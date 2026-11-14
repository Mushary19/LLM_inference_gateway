"""FakeProvider unit tests: no FastAPI, no database."""

import asyncio

import pytest

from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ModelNotFoundError,
    ProviderUnavailableError,
    Usage,
)
from app.providers.fake import FAKE_MODEL, FakeProvider


def request(*contents: str, model: str = FAKE_MODEL, max_tokens: int | None = None) -> ChatRequest:
    messages = tuple(ChatMessage(role="user", content=c) for c in contents)
    return ChatRequest(model=model, messages=messages, max_tokens=max_tokens)


async def collect(provider: FakeProvider, chat_request: ChatRequest):
    return [chunk async for chunk in provider.stream(chat_request)]


def test_chat_is_deterministic_and_answers_the_last_user_message():
    provider = FakeProvider()
    chat_request = ChatRequest(
        model=FAKE_MODEL,
        messages=(
            ChatMessage("system", "Be brief."),
            ChatMessage("user", "first question"),
            ChatMessage("assistant", "first answer"),
            ChatMessage("user", "What is KV caching?"),
        ),
    )

    first = asyncio.run(provider.chat(chat_request))
    second = asyncio.run(provider.chat(chat_request))

    assert first == second
    assert first.message == ChatMessage("assistant", "Fake response to: What is KV caching?")
    assert first.model == FAKE_MODEL and first.finish_reason == "stop"
    # Word counts: 2 + 2 + 2 + 4 prompt words, 7 reply words.
    assert first.usage == Usage(prompt_tokens=10, completion_tokens=7)
    assert first.usage.total_tokens == 17


def test_max_tokens_truncates_and_reports_length():
    response = asyncio.run(FakeProvider().chat(request("one two three", max_tokens=2)))

    assert response.message.content == "Fake response"
    assert response.finish_reason == "length"
    assert response.usage.completion_tokens == 2


def test_unknown_model_is_rejected():
    with pytest.raises(ModelNotFoundError):
        asyncio.run(FakeProvider().chat(request("hi", model="gpt-unknown")))


def test_injected_failure_is_raised_and_reported_by_health():
    provider = FakeProvider(fail_with=ProviderUnavailableError("upstream down"))

    with pytest.raises(ProviderUnavailableError, match="upstream down"):
        asyncio.run(provider.chat(request("hi")))
    health = asyncio.run(provider.health())
    assert not health.healthy and health.detail == "upstream down"


def test_stream_yields_the_same_text_as_chat_with_usage_on_the_final_chunk():
    provider = FakeProvider()
    chat_request = request("Explain KV caching")

    chunks = asyncio.run(collect(provider, chat_request))
    response = asyncio.run(provider.chat(chat_request))

    assert "".join(c.delta for c in chunks) == response.message.content
    assert all(c.finish_reason is None and c.usage is None for c in chunks[:-1])
    assert chunks[-1].finish_reason == "stop" and chunks[-1].usage == response.usage


def test_stream_raises_provider_errors_too():
    provider = FakeProvider(fail_with=ProviderUnavailableError("down"))

    with pytest.raises(ProviderUnavailableError):
        asyncio.run(collect(provider, request("hi")))


def test_list_models_and_health():
    provider = FakeProvider(models=("a", "b"))

    models = asyncio.run(provider.list_models())

    assert [(m.id, m.owned_by) for m in models] == [("a", "fake"), ("b", "fake")]
    assert asyncio.run(provider.health()).healthy
