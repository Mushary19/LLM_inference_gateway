"""Gateway endpoints: POST /v1/chat/completions and GET /v1/models.

Every /v1 route requires an API key (Authorization: Bearer <key>) and is rate-limited per
key (see app/api/v1.py). Both checks run before the route, so a rejected request never
reaches the provider. Chat completions also reserve token quota before calling the provider
and settle it afterwards (see app/services/quota.py). The routes call providers directly,
not the legacy /generate pipeline.

Routing: the request's public model ID is resolved in the model registry
(app/providers/registry.py) to a provider instance and that provider's own model ID. Clients
only ever see public IDs. Provider-specific model IDs, provider URLs and upstream error
details stay in the server logs.
"""

import logging
import re
import time
import uuid

from fastapi import Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.auth import AuthenticatedClient, require_api_key
from app.api.v1 import router
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
# fixed: upstream details (URLs, provider model IDs, response bodies) go to the logs only.
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


def _provider_http_error(route: ModelRoute, exc: ProviderError) -> HTTPException:
    status_code, message = next(
        ((code, text) for error_type, code, text in _ERROR_RESPONSES if isinstance(exc, error_type)),
        _DEFAULT_ERROR_RESPONSE,
    )
    if isinstance(exc, ModelNotFoundError):
        logger.error(
            "Provider %s does not serve model %r (public model %s); check the model registry: %s",
            route.provider.name, route.provider_model_id, route.public_id, exc,
        )
    else:
        logger.warning(
            "Provider %s failed for model %s (%s): %s", route.provider.name, route.public_id, type(exc).__name__, exc
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
) -> None:
    """Settle the reservation. If the database fails, keep the response: the provider has
    already done the work. The reservation stays `reserved` and holds its tokens until
    `python -m app.cli expire-reservations` settles it."""
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


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    client: AuthenticatedClient = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> ChatCompletionResponse:
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

    request_id = uuid.uuid4()
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

    # Provider call: no database transaction or connection is held here. A non-provider
    # exception (a bug) or cancellation leaves the reservation for expire-reservations.
    started = time.monotonic()
    try:
        response = await provider.chat(provider_request)
    except ProviderError as exc:
        await _settle(
            db, reservation, usage=exc.usage, succeeded=False,
            may_have_consumed_tokens=exc.may_have_consumed_tokens, model=route.public_id, started=started,
        )
        raise _provider_http_error(route, exc) from exc

    # Transaction 2.
    await _settle(
        db, reservation, usage=response.usage, succeeded=True,
        may_have_consumed_tokens=True, model=route.public_id, started=started,
    )
    return _to_http_response(response, request_id, route.public_id)


@router.get("/models", response_model=ModelList)
async def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> ModelList:
    """The registered, enabled public models. Providers are not asked: this lists what the
    gateway routes, not what each upstream happens to serve, so it works while they're down."""
    return ModelList(
        data=[ModelObject(id=route.public_id, owned_by=MODEL_OWNER) for route in registry.enabled_models()]
    )
