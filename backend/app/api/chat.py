"""Gateway endpoints: POST /v1/chat/completions and GET /v1/models.

Every /v1 route requires an API key (Authorization: Bearer <key>) and is rate-limited per
key (see app/api/v1.py). Both checks run before the route, so a rejected request never
reaches the provider. Chat completions also reserve token quota before calling the provider
and settle it afterwards (see app/services/quota.py). The routes call providers directly,
not the legacy /generate pipeline.

Routing: the request's public model ID is resolved in the model registry
(app/providers/registry.py) to a provider instance and that provider's own model ID. Clients
only ever see public IDs. Provider-specific model IDs, provider URLs and upstream error
details never reach clients. Provider failures are logged as structured metadata (error type,
upstream status), without the upstream error text (see inference_log.log_provider_failure).

Streaming ("stream": true) shares everything up to and including the quota reservation; the
streamed response and its settlement are in app/api/chat_stream.py.

Request IDs: the reservation, the usage record and `chatcmpl-<hex>` use the request's internal
gateway UUID (app/core/request_context.py), never the client's X-Request-ID. Every request
that gets as far as the provider logs one `inference_request_finished` line
(app/api/inference_log.py): from this module for non-streaming requests, from
StreamSettlement.finish() for streams.
"""

import logging
import re
import time
import uuid
from functools import partial

import anyio
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.auth import AuthenticatedClient, require_api_key
from app.api.chat_stream import GatewayStreamingResponse, StreamSettlement, sse_events, stream_meta
from app.api.inference_log import InferenceSummary, interrupted_outcome, log_provider_failure, provider_error_outcome
from app.api.v1 import router
from app.core import request_context
from app.core.config import settings
from app.database.connection import get_db
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    InvalidProviderRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.schemas.chat import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageOut,
    ChatUsage,
    ModelList,
    ModelObject,
)
from app.providers.registry import ModelRegistry, ModelRoute, UnknownModelError
from app.services import quota

logger = logging.getLogger(__name__)

# Most specific first; anything else from a provider is a bad gateway. Public messages are
# fixed: upstream details (URLs, provider model IDs, response bodies) never reach the client.
# A ModelNotFoundError here means the registry routed a public model to a provider that
# doesn't serve it, which is a gateway configuration problem, not a client error.
_ERROR_RESPONSES: list[tuple[type[ProviderError], int, str]] = [
    (ModelNotFoundError, status.HTTP_502_BAD_GATEWAY, "The provider could not serve the requested model."),
    (InvalidProviderRequestError, status.HTTP_400_BAD_REQUEST, "The provider rejected the request as invalid."),
    (ProviderTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT, "The provider did not respond in time."),
    (ProviderUnavailableError, status.HTTP_503_SERVICE_UNAVAILABLE, "The provider is temporarily unavailable."),
]
_DEFAULT_ERROR_RESPONSE = (status.HTTP_502_BAD_GATEWAY, "The provider returned an unexpected response.")

MODEL_NOT_FOUND = {"code": "model_not_found", "message": "The requested model was not found."}
MODEL_OWNER = "inferenceflow"

# Sanitizing upstream 400 messages, which clients need (e.g. "context length exceeded").
_ADAPTER_PREFIX = re.compile(r"^provider returned HTTP \d{3}:\s*")
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")
_CONTROL_AND_SPACE = re.compile(r"[\s\x00-\x1f\x7f]+")
_MAX_PUBLIC_MESSAGE_CHARS = 300


def get_model_registry(request: Request) -> ModelRegistry:
    """The registry built at startup. Tests override this dependency."""
    return request.app.state.model_registry


def sanitize_upstream_message(message: str, route: ModelRoute) -> str:
    """Make an upstream rejection message safe to show to the client.

    Drops the adapter's "provider returned HTTP nnn:" prefix, replaces the provider's model
    ID with the public one, redacts URLs and IP addresses, removes control characters and
    caps the length.
    """
    text = _ADAPTER_PREFIX.sub("", message)
    if route.provider_model_id != route.public_id:
        text = text.replace(route.provider_model_id, route.public_id)
    text = _URL.sub("[redacted]", text)
    text = _IPV4.sub("[redacted]", text)
    text = _CONTROL_AND_SPACE.sub(" ", text).strip()
    if len(text) > _MAX_PUBLIC_MESSAGE_CHARS:
        text = text[: _MAX_PUBLIC_MESSAGE_CHARS - 3].rstrip() + "..."
    return text


def _provider_http_error(route: ModelRoute, exc: ProviderError, *, stream: bool) -> HTTPException:
    status_code, message = next(
        ((code, text) for error_type, code, text in _ERROR_RESPONSES if isinstance(exc, error_type)),
        _DEFAULT_ERROR_RESPONSE,
    )
    log_provider_failure(
        provider=route.provider.name, model=route.public_id, provider_model_id=route.provider_model_id,
        stream=stream, phase="before_accept" if stream else "response", exc=exc,
    )
    if isinstance(exc, InvalidProviderRequestError):
        upstream = sanitize_upstream_message(str(exc), route)
        if upstream:
            message = f"The provider rejected the request: {upstream}"
    return HTTPException(status_code=status_code, detail=message)


def _to_provider_request(body: ChatCompletionRequest, route: ModelRoute) -> ChatRequest:
    return ChatRequest(
        model=route.provider_model_id,
        messages=tuple(ChatMessage(role=m.role, content=m.content) for m in body.messages),
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
    )


def _to_http_response(response: ChatResponse, request_id: uuid.UUID, public_model: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        # The gateway assigns the ID (its request ID, also used for quota and usage records)
        # and timestamp; providers stay deterministic.
        id=f"chatcmpl-{request_id.hex}",
        created=int(time.time()),
        # Always the public ID the client asked for, never the upstream's own model name.
        model=public_model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessageOut(role="assistant", content=response.message.content),
                finish_reason=response.finish_reason,
            )
        ],
        usage=None
        if response.usage is None
        else ChatUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        ),
    )


def _insufficient_quota(exc: quota.InsufficientQuotaError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "code": "insufficient_quota",
            "message": "Not enough token quota left for this request",
            "required_tokens": exc.required_tokens,
            "available_tokens": exc.available_tokens,
        },
    )


async def _settle(
    db: Session,
    reservation: quota.Reservation,
    *,
    usage: Usage | None,
    succeeded: bool,
    may_have_consumed_tokens: bool,
    model: str,
    started: float,
) -> bool:
    """Settle the reservation. If the database fails, keep the response: the provider has
    already done the work. The reservation stays `reserved` and holds its tokens until
    `python -m app.cli expire-reservations` settles it. Returns whether it settled (for the
    lifecycle log only)."""
    try:
        await run_in_threadpool(
            quota.settle,
            db,
            reservation.reservation_id,
            usage=usage,
            succeeded=succeeded,
            may_have_consumed_tokens=may_have_consumed_tokens,
            model=model,
            latency_ms=(time.monotonic() - started) * 1000,
        )
    except (SQLAlchemyError, quota.QuotaNotFoundError):
        logger.exception(
            "Could not settle quota reservation for request %s; it stays reserved until expired",
            reservation.request_id,
        )
        return False
    return True


async def _start_stream(
    db: Session,
    route: ModelRoute,
    provider_request: ChatRequest,
    reservation: quota.Reservation,
    *,
    request_id: uuid.UUID,
    include_usage: bool,
    started: float,
    response: Response,
    summary: InferenceSummary,
) -> GatewayStreamingResponse:
    """Open the provider stream and wait for its first item (the provider's "accepted"
    chunk), then hand it to a StreamSettlement and the streaming response.

    Until that first item, failures are settled as for non-streaming requests and returned
    as ordinary HTTP errors. Nothing watches the client during this wait (see chat_stream).
    """
    provider_stream = route.provider.stream(provider_request)
    owner = StreamSettlement(
        provider_stream,
        partial(_settle, db, reservation, model=route.public_id, started=started),
        request_id,
        summary,
    )
    try:
        first = await provider_stream.__anext__()
    except ProviderError as exc:
        http_error = _provider_http_error(route, exc, stream=True)
        await owner.finish(
            usage=exc.usage, succeeded=False, may_have_consumed_tokens=exc.may_have_consumed_tokens,
            outcome=provider_error_outcome(exc), http_status=http_error.status_code,
        )
        raise http_error from exc
    except StopAsyncIteration:
        # The provider ended without yielding anything: a contract violation, and the request
        # was sent, so it may have consumed tokens. Not a success, and not a bare 500.
        logger.error("Provider %s ended a stream without any chunk (model %s)", route.provider.name, route.public_id)
        http_error = _provider_http_error(route, ProviderError("provider stream ended without any chunk"), stream=True)
        await owner.finish(
            usage=None, succeeded=False, may_have_consumed_tokens=True,
            outcome="provider_error", http_status=http_error.status_code,
        )
        raise http_error from None
    except BaseException as exc:
        # A bug in the provider, or cancellation: close and settle conservatively, then re-raise.
        outcome = interrupted_outcome(exc)
        with anyio.CancelScope(shield=True):
            await owner.finish(
                usage=None, succeeded=False, may_have_consumed_tokens=True,
                outcome=outcome, http_status=500 if outcome == "gateway_error" else None,
            )
        raise

    # Headers set by dependencies (X-RateLimit-*) are only merged by FastAPI into responses it
    # builds itself, so they are copied onto the streaming response explicitly.
    headers = {
        key: value for key, value in response.headers.items() if key.lower() not in ("content-length", "content-type")
    }
    meta = stream_meta(request_id, route.public_id, include_usage)
    return GatewayStreamingResponse(
        sse_events(first, provider_stream, owner, meta, route), owner=owner, headers=headers
    )


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    registry: ModelRegistry = Depends(get_model_registry),
    client: AuthenticatedClient = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> ChatCompletionResponse | GatewayStreamingResponse:
    logger.info(
        "Chat completion for application %s (key %s), model %s",
        client.application_name, client.key_prefix, body.model,
    )
    # Resolved before any quota work: an unknown model creates no reservation or usage record.
    try:
        route = registry.resolve(body.model)
    except UnknownModelError as exc:
        logger.info("Rejected request for application %s: %s", client.application_name, exc)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=MODEL_NOT_FOUND) from exc

    # The internal UUID: the key of the reservation and usage record. Never the client's ID.
    request_id = request_context.gateway_request_id()
    provider = route.provider
    provider_request = _to_provider_request(body, route)
    estimate = quota.estimate_request_tokens(
        provider_request.messages, body.max_tokens or settings.max_generation_tokens
    )
    # Transaction 1 (commits, releasing the connection before the provider call).
    # Usage is recorded under the provider instance name and the public model ID.
    try:
        reservation = await run_in_threadpool(
            quota.reserve,
            db,
            application_id=client.application_id,
            api_key_id=client.api_key_id,
            request_id=request_id,
            tokens=estimate,
            provider=provider.name,
            model=route.public_id,
        )
    except quota.InsufficientQuotaError as exc:
        logger.info("Insufficient quota for application %s (key %s)", client.application_name, client.key_prefix)
        raise _insufficient_quota(exc) from exc
    except SQLAlchemyError as exc:
        logger.exception("Could not reserve quota for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quota accounting is temporarily unavailable",
        ) from exc

    started = time.monotonic()
    summary = InferenceSummary(provider=provider.name, model=route.public_id, stream=body.stream, started=started)
    if body.stream:
        return await _start_stream(
            db, route, provider_request, reservation,
            request_id=request_id,
            include_usage=body.stream_options is not None and body.stream_options.include_usage,
            started=started,
            response=response,
            summary=summary,
        )

    try:
        # Provider call: no database transaction or connection is held here. A non-provider
        # exception (a bug) or cancellation leaves the reservation for expire-reservations.
        try:
            completion = await provider.chat(provider_request)
        except ProviderError as exc:
            summary.settlement = "unknown"
            settled = await _settle(
                db, reservation, usage=exc.usage, succeeded=False,
                may_have_consumed_tokens=exc.may_have_consumed_tokens, model=route.public_id, started=started,
            )
            summary.settlement = "succeeded" if settled else "failed"
            http_error = _provider_http_error(route, exc, stream=False)
            summary.finish(outcome=provider_error_outcome(exc), http_status=http_error.status_code, usage=exc.usage)
            raise http_error from exc

        # Transaction 2.
        summary.settlement = "unknown"
        settled = await _settle(
            db, reservation, usage=completion.usage, succeeded=True,
            may_have_consumed_tokens=True, model=route.public_id, started=started,
        )
        summary.settlement = "succeeded" if settled else "failed"
        result = _to_http_response(completion, request_id, route.public_id)
        summary.finish(outcome="completed", http_status=status.HTTP_200_OK, usage=completion.usage)
        return result
    except BaseException as exc:
        summary.finish_interrupted(exc)  # no-op if the outcome was already logged
        raise


@router.get("/models", response_model=ModelList)
async def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> ModelList:
    """The registered, enabled public models. Providers are not asked: this lists what the
    gateway routes, not what each upstream happens to serve, so it works while they're down."""
    return ModelList(
        data=[ModelObject(id=route.public_id, owned_by=MODEL_OWNER) for route in registry.enabled_models()]
    )
