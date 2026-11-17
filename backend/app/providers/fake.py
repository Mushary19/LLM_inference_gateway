"""Deterministic in-process provider for tests and local development. Not a language model.

Successor of SimulatedInferenceEngine for the gateway API: same idea (optional delay,
fixed output, failure injection), but speaking the normalized chat types.
"""

import asyncio
from collections.abc import AsyncIterator

from app.providers.base import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    HealthStatus,
    ModelInfo,
    ModelNotFoundError,
    ProviderError,
    Usage,
)

FAKE_MODEL = "fake-model"


class FakeProvider:
    """Replies "Fake response to: <last user message>".

    Token counts are whitespace word counts, not real tokens. `max_tokens` truncates the
    reply to that many words and reports finish_reason="length".
    """

    name = "fake"  # default instance name; the adapter type is also "fake"

    def __init__(
        self,
        models: tuple[str, ...] = (FAKE_MODEL,),
        delay_seconds: float = 0.0,
        fail_with: ProviderError | None = None,
        *,
        name: str = "fake",
    ) -> None:
        self.name = name
        self.models = models
        self.delay_seconds = delay_seconds
        self.fail_with = fail_with

    def __repr__(self) -> str:
        return f"FakeProvider(name={self.name!r})"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        words, finish_reason = await self._complete(request)
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=" ".join(words)),
            finish_reason=finish_reason,
            usage=self._usage(request, words),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        words, finish_reason = await self._complete(request)
        for i, word in enumerate(words):
            yield ChatChunk(delta=word if i == 0 else f" {word}")
        yield ChatChunk(delta="", finish_reason=finish_reason, usage=self._usage(request, words))

    async def list_models(self) -> list[ModelInfo]:
        if self.fail_with is not None:  # consistent with health(): a failing provider can't list
            raise self.fail_with
        return [ModelInfo(id=model, owned_by=self.name) for model in self.models]

    async def health(self) -> HealthStatus:
        if self.fail_with is not None:
            return HealthStatus(healthy=False, detail=str(self.fail_with))
        return HealthStatus(healthy=True)

    async def aclose(self) -> None:
        pass

    async def _complete(self, request: ChatRequest) -> tuple[list[str], FinishReason]:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.fail_with is not None:
            raise self.fail_with
        if request.model not in self.models:
            raise ModelNotFoundError(f"model {request.model!r} is not available")
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        words = f"Fake response to: {last_user}".split()
        if request.max_tokens is not None and len(words) > request.max_tokens:
            return words[: request.max_tokens], "length"
        return words, "stop"

    @staticmethod
    def _usage(request: ChatRequest, completion_words: list[str]) -> Usage:
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        return Usage(prompt_tokens=prompt_tokens, completion_tokens=len(completion_words))
