"""Streaming responses for POST /v1/chat/completions with "stream": true (Server-Sent Events).

Lifecycle (the route in app/api/chat.py does steps 1-3):
1. Quota is reserved exactly as for non-streaming requests (committed, no connection held).
2. The route opens the provider stream and waits for its first item only: the provider's
   "accepted" chunk. Failures until then are ordinary HTTP errors with the usual settlement.
3. The route hands the provider stream to a StreamSettlement and returns a
   GatewayStreamingResponse. From here on the HTTP status is 200.
4. sse_events() sends the role chunk, content chunks, the finish chunk, the usage chunk (only
   if the client asked for it), settles the reservation, and only then sends [DONE]. A failure
   after the stream started is sent as one error event, and the stream ends without [DONE].

StreamSettlement is the single owner of the provider stream and the reservation: finish()
closes the stream and settles, once, shielded from cancellation. Every way a response can end
reaches it (normal end, provider failure, a gateway bug, client disconnect, cancellation, a
failing ASGI send, or a body that never started) because GatewayStreamingResponse calls it
in a finally block; sse_events() calls it first on the paths where it knows the outcome.

anyio cancel scopes are only ever opened and closed within one step: the generators here are
started in the route's task, continued in Starlette's streaming task and may be closed from
the response's task, and a scope may not cross tasks.

Known limitation: while the route waits for the provider's accepted chunk (step 2), nothing
watches the client connection, because no response exists yet. A client that disconnects
then is noticed as soon as the response starts; the upstream request is aborted and the
reservation settled at that point.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from app.providers.base import ChatChunk, ProviderError, ProviderTimeoutError, ProviderUnavailableError, Usage
from app.providers.registry import ModelRoute
from app.schemas.chat import (
    ChatCompletionChunk,
    ChatUsage,
    ChunkChoice,
    ChunkDelta,
    StreamError,
    StreamErrorEvent,
)

logger = logging.getLogger(__name__)

MEDIA_TYPE = "text/event-stream"
# no-cache: never store a stream; X-Accel-Buffering: tell nginx-style proxies not to buffer it.
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
DONE = b"data: [DONE]\n\n"

# Public error messages are the same fixed texts as the non-streaming HTTP errors.
_PROVIDER_ERROR_EVENTS: list[tuple[type[ProviderError], str, str]] = [
    (ProviderTimeoutError, "provider_timeout", "The provider did not respond in time."),
    (ProviderUnavailableError, "provider_unavailable", "The provider is temporarily unavailable."),
]
_DEFAULT_PROVIDER_ERROR_EVENT = ("provider_error", "The provider returned an unexpected response.")
INTERNAL_ERROR = StreamError(message="An internal error occurred.", type="server_error", code="internal_error")

# finish(usage, succeeded, may_have_consumed_tokens): settles the request's reservation.
SettleFunction = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class StreamMeta:
    request_id: uuid.UUID
    public_model: str
    include_usage: bool
    created: int

    @property
    def completion_id(self) -> str:
        return f"chatcmpl-{self.request_id.hex}"


class StreamSettlement:
    """Owns one streamed request's provider stream and reservation.

    finish() closes the provider stream and then settles the reservation, the first time
    it is called; later calls do nothing. Both steps are shielded from cancellation, so a
    disconnect or cancellation arriving meanwhile can't skip them. Settlement itself is
    idempotent in the database too (quota.settle checks the reservation under a row lock),
    so even a second settle call could not charge twice.

    Settlement has no timeout: the database call runs in a worker thread that can't be
    interrupted, so a timeout would only hide whether it committed.
    """

    def __init__(self, provider_stream: AsyncIterator[ChatChunk], settle: SettleFunction, request_id: uuid.UUID) -> None:
        self._stream = provider_stream
        self._settle = settle
        self._request_id = request_id
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    async def finish(self, *, usage: Usage | None, succeeded: bool, may_have_consumed_tokens: bool) -> None:
        if self._finished:
            return
        self._finished = True  # before any await: a re-entrant call can't settle again
        with anyio.CancelScope(shield=True):
            try:
                await self._stream.aclose()  # no-op if the stream already ended or raised
            except Exception:
                logger.exception("Could not close the provider stream for request %s", self._request_id)
            finally:
                try:
                    await self._settle(
                        usage=usage, succeeded=succeeded, may_have_consumed_tokens=may_have_consumed_tokens
                    )
                except Exception:
                    # Database failures are already handled (and logged) by the settle function;
                    # this is for anything else. The reservation stays for expire-reservations.
                    logger.exception(
                        "Could not settle the reservation for request %s; it stays reserved until expired",
                        self._request_id,
                    )


def encode(chunk: ChatCompletionChunk | StreamErrorEvent) -> bytes:
    payload = chunk.model_dump(mode="json", exclude_unset=True)
    return b"data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"


def _chunk(meta: StreamMeta, choices: list[ChunkChoice], **extra) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=meta.completion_id,
        object="chat.completion.chunk",
        created=meta.created,
        model=meta.public_model,  # always the public ID, never the upstream's model name
        choices=choices,
        **extra,
    )


def role_chunk(meta: StreamMeta) -> ChatCompletionChunk:
    return _chunk(meta, [ChunkChoice(index=0, delta=ChunkDelta(role="assistant", content=""), finish_reason=None)])


def content_chunk(meta: StreamMeta, text: str) -> ChatCompletionChunk:
    return _chunk(meta, [ChunkChoice(index=0, delta=ChunkDelta(content=text), finish_reason=None)])


def finish_chunk(meta: StreamMeta, finish_reason: str) -> ChatCompletionChunk:
    return _chunk(meta, [ChunkChoice(index=0, delta=ChunkDelta(), finish_reason=finish_reason)])


def usage_chunk(meta: StreamMeta, usage: Usage | None) -> ChatCompletionChunk:
    reported = (
        None
        if usage is None
        else ChatUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
    )
    return _chunk(meta, [], usage=reported)


def provider_error_event(exc: ProviderError) -> StreamErrorEvent:
    code, message = next(
        ((code, text) for error_type, code, text in _PROVIDER_ERROR_EVENTS if isinstance(exc, error_type)),
        _DEFAULT_PROVIDER_ERROR_EVENT,
    )
    return StreamErrorEvent(error=StreamError(message=message, type="provider_error", code=code))


async def sse_events(
    first: ChatChunk,
    provider_stream: AsyncIterator[ChatChunk],
    owner: StreamSettlement,
    meta: StreamMeta,
    route: ModelRoute,
) -> AsyncIterator[bytes]:
    """The response body. Never closes the provider stream itself: the owner does.

    Cancellation and GeneratorExit are not caught here; GatewayStreamingResponse settles them.
    """
    yield encode(role_chunk(meta))
    chunk = first
    try:
        while True:
            if chunk.delta:
                yield encode(content_chunk(meta, chunk.delta))
            if chunk.finish_reason is not None:
                break
            try:
                chunk = await provider_stream.__anext__()
            except StopAsyncIteration:
                # The provider broke its contract (no terminal chunk): incomplete, not a success.
                raise ProviderError("provider stream ended without a finish chunk") from None
    except ProviderError as exc:
        logger.warning(
            "Provider %s failed mid-stream for model %s, request %s (%s): %s",
            route.provider.name, route.public_id, meta.request_id, type(exc).__name__, exc,
        )
        await owner.finish(usage=exc.usage, succeeded=False, may_have_consumed_tokens=True)
        yield encode(provider_error_event(exc))
        return
    except Exception:
        logger.exception("Streaming failed for request %s", meta.request_id)
        await owner.finish(usage=None, succeeded=False, may_have_consumed_tokens=True)
        yield encode(StreamErrorEvent(error=INTERNAL_ERROR))
        return

    yield encode(finish_chunk(meta, chunk.finish_reason))
    if meta.include_usage:
        yield encode(usage_chunk(meta, chunk.usage))
    # Settled before [DONE], so [DONE] means the request is complete and accounted for. If the
    # database fails, the settle function logs it and leaves the reservation for expiry; the
    # client still gets [DONE] because it has its whole output.
    await owner.finish(usage=chunk.usage, succeeded=True, may_have_consumed_tokens=True)
    yield DONE


class GatewayStreamingResponse(StreamingResponse):
    """A StreamingResponse that always closes its body and finishes its StreamSettlement.

    Starlette doesn't close the body iterator when the response ends early (a disconnect
    cancels the streaming task, possibly while it is inside ASGI send(); a failing send
    raises out of it). The finally block here covers all of those, and a body that never
    started. Anything not settled by then is settled as "may have consumed tokens", which
    charges the full reservation unless usage was reported, as for other failures.
    """

    def __init__(self, content: AsyncIterator[bytes], *, owner: StreamSettlement, headers: dict[str, str]) -> None:
        super().__init__(content, media_type=MEDIA_TYPE, headers={**headers, **STREAM_HEADERS})
        self._owner = owner

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()  # type: ignore[attr-defined]
                except Exception:
                    logger.exception("Could not close the response body")
            if not self._owner.finished:
                logger.info("Stream ended early (client disconnected or the response was cancelled)")
            await self._owner.finish(usage=None, succeeded=False, may_have_consumed_tokens=True)


def stream_meta(request_id: uuid.UUID, public_model: str, include_usage: bool) -> StreamMeta:
    return StreamMeta(request_id=request_id, public_model=public_model, include_usage=include_usage, created=int(time.time()))
