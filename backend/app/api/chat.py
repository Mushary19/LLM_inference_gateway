"""Gateway endpoints: POST /v1/chat/completions and GET /v1/models.

Every /v1 route requires an API key (Authorization: Bearer <key>); authentication runs
before the route, so a rejected request never reaches the provider. The routes call the
configured provider directly, not the legacy /generate pipeline.
"""

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.auth import AuthenticatedClient, require_api_key
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

logger = logging.getLogger(__name__)

# Router-level dependency: every route added here is authenticated by default. Routes that
# need the client also declare require_api_key; FastAPI resolves it once per request.
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

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


def _to_http_response(response: ChatResponse) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        # The gateway assigns the ID and timestamp; providers stay deterministic.
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=response.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessageOut(role="assistant", content=response.message.content),
                finish_reason=response.finish_reason,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        ),
    )


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    provider: Provider = Depends(get_chat_provider),
    client: AuthenticatedClient = Depends(require_api_key),
) -> ChatCompletionResponse:
    logger.info(
        "Chat completion for application %s (key %s), model %s",
        client.application_name, client.key_prefix, body.model,
    )
    try:
        response = await provider.chat(_to_provider_request(body))
    except ProviderError as exc:
        raise _provider_http_error(provider, exc)
    return _to_http_response(response)


@router.get("/models", response_model=ModelList)
async def list_models(provider: Provider = Depends(get_chat_provider)) -> ModelList:
    try:
        models = await provider.list_models()
    except ProviderError as exc:
        raise _provider_http_error(provider, exc)
    return ModelList(data=[ModelObject(id=m.id, owned_by=m.owned_by) for m in models])
