"""Gateway endpoints: POST /v1/chat/completions and GET /v1/models.

Every /v1 route requires an API key (Authorization: Bearer <key>) and is rate-limited per
key (see app/api/v1.py). Both checks run before the route, so a rejected request never
reaches the provider. Chat completions also reserve token quota before calling the provider
and settle it afterwards (see app/services/quota.py). The routes call the configured
provider directly, not the legacy /generate pipeline.
"""

import logging
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
    Provider,
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
from app.services import quota

logger = logging.getLogger(__name__)

# Most specific first; anything else from a provider is a bad gateway.
_ERROR_STATUS: list[tuple[type[ProviderError], int]] = [
    (ModelNotFoundError, status.HTTP_404_NOT_FOUND),
    (InvalidProviderRequestError, status.HTTP_400_BAD_REQUEST),
    (ProviderTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT),
    (ProviderUnavailableError, status.HTTP_503_SERVICE_UNAVAILABLE),
]


def get_chat_provider(request: Request) -> Provider:
    """The provider created at startup. Tests override this dependency."""
    return request.app.state.chat_provider


def _provider_http_error(provider: Provider, exc: ProviderError) -> HTTPException:
    status_code = next(
        (code for error_type, code in _ERROR_STATUS if isinstance(exc, error_type)),
        status.HTTP_502_BAD_GATEWAY,
    )
    logger.warning("Provider %s failed (%s): %s", provider.name, type(exc).__name__, exc)
    return HTTPException(status_code=status_code, detail=str(exc) or type(exc).__name__)


def _to_provider_request(body: ChatCompletionRequest) -> ChatRequest:
    return ChatRequest(
        model=body.model,
        messages=tuple(ChatMessage(role=m.role, content=m.content) for m in body.messages),
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
    )


def _to_http_response(response: ChatResponse, request_id: uuid.UUID) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        # The gateway assigns the ID (its request ID, also used for quota and usage records)
        # and timestamp; providers stay deterministic.
        id=f"chatcmpl-{request_id.hex}",
        created=int(time.time()),
        model=response.model,
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
    provider: Provider = Depends(get_chat_provider),
    client: AuthenticatedClient = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> ChatCompletionResponse:
    logger.info(
        "Chat completion for application %s (key %s), model %s",
        client.application_name, client.key_prefix, body.model,
    )
    request_id = uuid.uuid4()
    provider_request = _to_provider_request(body)
    estimate = quota.estimate_request_tokens(
        provider_request.messages, body.max_tokens or settings.max_generation_tokens
    )
    # Transaction 1 (commits, releasing the connection before the provider call).
    try:
        reservation = await run_in_threadpool(
            quota.reserve,
            db,
            application_id=client.application_id,
            api_key_id=client.api_key_id,
            request_id=request_id,
            tokens=estimate,
            provider=provider.name,
            model=body.model,
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
            may_have_consumed_tokens=exc.may_have_consumed_tokens, model=body.model, started=started,
        )
        raise _provider_http_error(provider, exc)

    # Transaction 2.
    await _settle(
        db, reservation, usage=response.usage, succeeded=True,
        may_have_consumed_tokens=True, model=response.model, started=started,
    )
    return _to_http_response(response, request_id)


@router.get("/models", response_model=ModelList)
async def list_models(provider: Provider = Depends(get_chat_provider)) -> ModelList:
    try:
        models = await provider.list_models()
    except ProviderError as exc:
        raise _provider_http_error(provider, exc)
    return ModelList(data=[ModelObject(id=m.id, owned_by=m.owned_by) for m in models])
