"""Provider for any server that implements the OpenAI chat completions HTTP API.

Covers Ollama (http://localhost:11434/v1), vLLM's OpenAI-compatible server, and hosted
OpenAI-compatible APIs. Only non-streaming chat, model listing and a health check are
implemented so far.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

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

logger = logging.getLogger(__name__)

_UNAVAILABLE_STATUSES = {429, 502, 503}
_TIMEOUT_STATUSES = {408, 504}


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
    ) -> None:
        self.name = name
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
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
        }
        for field in ("max_tokens", "temperature", "top_p"):
            value = getattr(request, field)
            if value is not None:
                payload[field] = value

        body = await self._request_json("POST", "chat/completions", json=payload)
        return self._parse_chat_response(body, request.model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        raise NotImplementedError("streaming is not implemented for this provider yet")
        yield  # makes this an async generator, matching the Provider interface

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
        except httpx.TimeoutException as exc:
            never_sent = isinstance(exc, (httpx.ConnectTimeout, httpx.PoolTimeout))
            raise ProviderTimeoutError(
                f"provider did not respond in time ({type(exc).__name__})",
                may_have_consumed_tokens=not never_sent,
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"could not reach provider: {exc}",
                may_have_consumed_tokens=not isinstance(exc, httpx.ConnectError),
            ) from exc

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
            return ModelNotFoundError(detail)
        if status in _TIMEOUT_STATUSES:
            return ProviderTimeoutError(detail, may_have_consumed_tokens=consumed)
        if status in _UNAVAILABLE_STATUSES:
            return ProviderUnavailableError(detail, may_have_consumed_tokens=consumed)
        if status in (400, 413, 422):
            return InvalidProviderRequestError(detail)
        # 401/403 (gateway misconfigured), other 4xx and 5xx: the gateway can't fix these.
        return ProviderError(detail, may_have_consumed_tokens=consumed)

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
