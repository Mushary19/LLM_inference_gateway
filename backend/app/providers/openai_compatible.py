"""Provider for any server that implements the OpenAI chat completions HTTP API.

Covers Ollama (http://localhost:11434/v1), vLLM's OpenAI-compatible server, and hosted
OpenAI-compatible APIs: chat (non-streaming and streaming), model listing and a health check.
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing, contextmanager
from typing import Any, Iterator

import anyio
import httpx2 as httpx

from app.providers.base import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    HealthStatus,
    InvalidProviderRequestError,
    ModelInfo,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.providers.sse import MAX_EVENT_BYTES, SSEDecoder, SSEEventTooLargeError

logger = logging.getLogger(__name__)

_UNAVAILABLE_STATUSES = {429, 502, 503}
_TIMEOUT_STATUSES = {408, 504}
# Enough of an error body for its message; the rest is not read.
_MAX_ERROR_BODY_BYTES = 64 * 1024


class OpenAICompatibleProvider:
    # Default instance name. Each configured instance (e.g. "groq", "ollama") gets its own name,
    # which is what usage records store; the adapter type stays "openai_compatible".
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        name: str = "openai_compatible",
        stream_usage: bool = False,
    ) -> None:
        self.name = name
        self._timeout_seconds = timeout_seconds
        # Ask for usage on streams (stream_options.include_usage). Off by default: not every
        # OpenAI-compatible server accepts the option, and a server that rejects it would
        # fail every streamed request. Enable it per provider once verified.
        self._stream_usage = stream_usage
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # One client for the provider's lifetime, so connections are pooled and reused.
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )

    def __repr__(self) -> str:
        # Never include the base URL or the API key.
        return f"OpenAICompatibleProvider(name={self.name!r})"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        body = await self._request_json("POST", "chat/completions", json=_chat_payload(request, stream=False))
        return self._parse_chat_response(body, request.model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Stream a chat completion (see Provider.stream for the contract).

        Yields an empty "accepted" chunk once the upstream has answered 200 with an event
        stream, then content chunks, then one terminal chunk. timeout_seconds bounds the
        whole stream (connecting, waiting and reading), not just each read: the deadline is
        re-armed around every await rather than held open across yields.
        """
        payload = _chat_payload(request, stream=True)
        if self._stream_usage:
            payload["stream_options"] = {"include_usage": True}
        deadline = anyio.current_time() + self._timeout_seconds
        http_request = self._client.build_request("POST", "chat/completions", json=payload)
        try:
            with _until(deadline):
                response = await self._client.send(http_request, stream=True)
        except TimeoutError as exc:
            raise ProviderTimeoutError("provider did not respond in time (deadline)") from exc
        except httpx.TransportError as exc:
            raise _transport_error(exc) from exc

        byte_stream = response.aiter_bytes()
        try:
            if response.status_code >= 400:
                raise self._error_for_status(await _read_error_response(response, deadline))
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type != "text/event-stream":
                raise ProviderError(f"provider answered a stream request with content type {content_type!r}")
            yield ChatChunk(delta="")  # accepted: the response can start
            async with aclosing(_parse_chat_stream(byte_stream, deadline, request.model)) as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            # Runs on completion, failure, aclose() and cancellation. Closing an unfinished
            # response drops its connection, which tells the upstream to stop generating.
            with anyio.CancelScope(shield=True):
                await byte_stream.aclose()
                await response.aclose()

    async def list_models(self) -> list[ModelInfo]:
        body = await self._request_json("GET", "models")
        try:
            return [
                ModelInfo(id=str(item["id"]), owned_by=str(item.get("owned_by", "")))
                for item in body["data"]
            ]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"invalid models response from provider: {exc!r}") from exc

    async def health(self) -> HealthStatus:
        try:
            await self.list_models()
        except ProviderError as exc:
            return HealthStatus(healthy=False, detail=str(exc))
        return HealthStatus(healthy=True)

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send a request and return the decoded JSON body, or raise a normalized ProviderError."""
        # may_have_consumed_tokens: False only when the request certainly never ran (it was
        # never sent, or the provider answered with an error); otherwise work may have happened.
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise _transport_error(exc) from exc

        if response.status_code >= 400:
            raise self._error_for_status(response)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError("provider returned a response that is not valid JSON") from exc

    @staticmethod
    def _error_for_status(response: httpx.Response) -> ProviderError:
        status = response.status_code
        code, message = _error_details(response)
        detail = f"provider returned HTTP {status}: {message}"
        # A 4xx means the provider refused the request. A 5xx may come from a proxy after the
        # model already ran, so it is treated as possibly consuming tokens.
        consumed = status >= 500
        if status == 404 or code == "model_not_found":
            return ModelNotFoundError(detail, status_code=status)
        if status in _TIMEOUT_STATUSES:
            return ProviderTimeoutError(detail, may_have_consumed_tokens=consumed, status_code=status)
        if status in _UNAVAILABLE_STATUSES:
            return ProviderUnavailableError(detail, may_have_consumed_tokens=consumed, status_code=status)
        if status in (400, 413, 422):
            return InvalidProviderRequestError(detail, status_code=status)
        # 401/403 (gateway misconfigured), other 4xx and 5xx: the gateway can't fix these.
        return ProviderError(detail, may_have_consumed_tokens=consumed, status_code=status)

    @staticmethod
    def _parse_chat_response(body: Any, requested_model: str) -> ChatResponse:
        try:
            choice = body["choices"][0]
            content = choice["message"].get("content") or ""
            model = body.get("model") or requested_model
            usage = _parse_usage(body.get("usage"))
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            # The provider answered 200, so the model probably ran even though we can't read it.
            raise ProviderError(f"invalid chat completion response from provider: {exc!r}") from exc

        if usage is None:
            logger.warning("Provider response for %s has no usage; token usage is unknown", model)
        return ChatResponse(
            model=str(model),
            message=ChatMessage(role="assistant", content=str(content)),
            finish_reason=_finish_reason(choice.get("finish_reason")),
            usage=usage,
        )


def _chat_payload(request: ChatRequest, *, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stream": stream,
    }
    for field in ("max_tokens", "temperature", "top_p"):
        value = getattr(request, field)
        if value is not None:
            payload[field] = value
    return payload


def _transport_error(exc: httpx.TransportError) -> ProviderError:
    """Normalize an httpx transport failure. may_have_consumed_tokens is False only when the
    request certainly never ran (never sent, or the connection was never established)."""
    if isinstance(exc, httpx.TimeoutException):
        never_sent = isinstance(exc, (httpx.ConnectTimeout, httpx.PoolTimeout))
        return ProviderTimeoutError(
            f"provider did not respond in time ({type(exc).__name__})",
            may_have_consumed_tokens=not never_sent,
        )
    return ProviderUnavailableError(
        f"could not reach provider: {exc}",
        may_have_consumed_tokens=not isinstance(exc, httpx.ConnectError),
    )


@contextmanager
def _until(deadline: float) -> Iterator[None]:
    """A timeout for one await, ending at `deadline`. Raises TimeoutError when it passes.

    Entered and left within a single step of the stream generator, never across a yield.
    """
    with anyio.fail_after(max(deadline - anyio.current_time(), 0)):
        yield


async def _read_error_response(response: httpx.Response, deadline: float) -> httpx.Response:
    """Read (the start of) an error response's body, so its message can be extracted."""
    body = bytearray()
    parts = response.aiter_bytes()
    try:
        while len(body) < _MAX_ERROR_BODY_BYTES:
            with _until(deadline):
                body += await parts.__anext__()
    except (StopAsyncIteration, TimeoutError, httpx.TransportError):
        pass  # whatever was read is enough: the status alone classifies the error
    finally:
        await parts.aclose()
    headers = [(k, v) for k, v in response.headers.items() if k.lower() not in ("content-encoding", "content-length")]
    return httpx.Response(
        response.status_code, headers=headers, content=bytes(body[:_MAX_ERROR_BODY_BYTES]), request=response.request
    )


async def _parse_chat_stream(byte_stream: AsyncIterator[bytes], deadline: float, model: str) -> AsyncIterator[ChatChunk]:
    """Turn the upstream event stream into content chunks and one terminal chunk.

    Complete means a finish_reason was received. Usage (top-level "usage", else Groq's
    "x_groq.usage") may follow the finish_reason; it is optional and never fatal. After the
    finish_reason, [DONE], EOF, a read failure or the deadline all end the stream as
    complete. Before it, they are failures (except [DONE] and EOF: incomplete stream).
    """
    decoder = SSEDecoder(MAX_EVENT_BYTES)
    finish_reason: FinishReason | None = None
    usage_raw: Any = None
    groq_usage_raw: Any = None
    done = False
    while not done:
        try:
            with _until(deadline):
                raw = await byte_stream.__anext__()
        except StopAsyncIteration:
            decoder.close()
            break
        except (TimeoutError, httpx.TransportError) as exc:
            if finish_reason is not None:
                logger.warning("Provider stream for %s did not close after finishing (%s)", model, type(exc).__name__)
                break
            if isinstance(exc, TimeoutError):
                raise ProviderTimeoutError("provider stream exceeded its deadline", may_have_consumed_tokens=True) from exc
            error = _transport_error(exc)
            error.may_have_consumed_tokens = True  # the request was already accepted
            raise error from exc
        try:
            events = decoder.feed(raw)
        except SSEEventTooLargeError as exc:
            raise ProviderError(str(exc)) from exc
        for data in events:
            if data == "[DONE]":
                done = True
                break
            if not data.strip():
                continue  # an empty event carries nothing
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise ProviderError("provider stream contained an event that is not valid JSON") from exc
            if not isinstance(event, dict):
                raise ProviderError("provider stream contained an event that is not a JSON object")
            if event.get("error"):
                raise ProviderError(f"provider sent an error event: {_event_error_message(event['error'])}")
            if event.get("usage") is not None:
                usage_raw = event["usage"]
            x_groq = event.get("x_groq")
            if isinstance(x_groq, dict) and x_groq.get("usage") is not None:
                groq_usage_raw = x_groq["usage"]
            choices = event.get("choices")
            if choices is None:
                continue
            if not isinstance(choices, list):
                raise ProviderError("provider stream contained an event with invalid choices")
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta")
                content = delta.get("content") if isinstance(delta, dict) else None
                # Reasoning fields ("reasoning", "reasoning_content") are deliberately dropped.
                if isinstance(content, str) and content and finish_reason is None:
                    yield ChatChunk(delta=content)
                if choice.get("finish_reason") is not None and finish_reason is None:
                    finish_reason = _finish_reason(choice["finish_reason"])
    if finish_reason is None:
        raise ProviderError("provider stream ended before a finish reason (incomplete)")
    usage = _stream_usage(usage_raw if usage_raw is not None else groq_usage_raw, model)
    yield ChatChunk(delta="", finish_reason=finish_reason, usage=usage)


def _stream_usage(raw: Any, model: str) -> Usage | None:
    """Usage from a stream, or None. Invalid usage is logged and treated as unreported: the
    client already has its output, and accounting then charges the full reservation."""
    if raw is None:
        logger.warning("Provider stream for %s reported no usage; token usage is unknown", model)
        return None
    try:
        return _parse_usage(raw)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        logger.warning("Provider stream for %s reported invalid usage (%s); treating it as unreported", model, exc)
        return None


def _event_error_message(error: Any) -> str:
    if isinstance(error, dict):
        error = error.get("message") or error.get("code") or "error"
    return str(error)[:200]


def _parse_usage(raw: Any) -> Usage | None:
    """None when the provider sent no usage; an error when it sent usage we can't trust."""
    if raw is None:
        return None
    prompt_tokens, completion_tokens = raw["prompt_tokens"], raw["completion_tokens"]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (prompt_tokens, completion_tokens)):
        raise ValueError(f"invalid usage counts {prompt_tokens!r}, {completion_tokens!r}")
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _finish_reason(value: Any) -> FinishReason:
    # Only "stop" and "length" exist in the normalized types; tool calls and content
    # filtering aren't supported yet, so anything else is reported as "stop".
    return "length" if value == "length" else "stop"


def _error_details(response: httpx.Response) -> tuple[str | None, str]:
    """Extract (code, message) from an OpenAI-style error body: {"error": {"message", "code"}}."""
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict):
        return error.get("code"), str(error.get("message") or response.reason_phrase)
    if isinstance(error, str):
        return None, error
    return None, response.text[:200] or response.reason_phrase
