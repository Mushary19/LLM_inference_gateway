"""Normalized chat types, the Provider interface, and the provider error hierarchy."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ChatResponse:
    model: str
    message: ChatMessage
    finish_reason: FinishReason
    # None when the provider didn't report token usage. Never filled with guesses.
    usage: Usage | None


@dataclass(frozen=True)
class ChatChunk:
    """One streamed piece of a response. The final chunk carries finish_reason and usage."""

    delta: str
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    owned_by: str


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    detail: str | None = None


class ProviderError(Exception):
    """Base class for failures reported by a provider.

    `may_have_consumed_tokens` tells quota accounting whether the provider might have done
    (and billed) work before failing; adapters set it per failure, otherwise each class
    has a conservative default. `usage` is set when the provider reported usage anyway.
    `status_code` is the upstream HTTP status when the failure was an HTTP error response,
    None otherwise (transport failures, unreadable responses).
    """

    default_may_have_consumed_tokens = True

    def __init__(
        self,
        message: str = "",
        *,
        may_have_consumed_tokens: bool | None = None,
        usage: Usage | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.may_have_consumed_tokens = (
            self.default_may_have_consumed_tokens
            if may_have_consumed_tokens is None
            else may_have_consumed_tokens
        )
        self.usage = usage


class ProviderUnavailableError(ProviderError):
    """The provider can't serve requests right now (down, overloaded, unreachable)."""


class ProviderTimeoutError(ProviderError):
    """The provider did not respond in time."""


class ModelNotFoundError(ProviderError):
    """The requested model is not served by this provider."""

    default_may_have_consumed_tokens = False


class InvalidProviderRequestError(ProviderError):
    """The provider rejected the request as invalid (e.g. context too long)."""

    default_may_have_consumed_tokens = False


class Provider(Protocol):
    name: str

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Implemented as an async generator: `async for chunk in provider.stream(request)`."""
        ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def health(self) -> HealthStatus: ...

    async def aclose(self) -> None:
        """Release resources such as HTTP connections. Called once at application shutdown."""
        ...
