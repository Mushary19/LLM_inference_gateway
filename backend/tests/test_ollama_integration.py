"""Opt-in integration test against a real Ollama server.

Skipped unless OLLAMA_INTEGRATION_MODEL is set to a model that has been pulled, e.g.:

    ollama pull qwen2.5:0.5b
    OLLAMA_INTEGRATION_MODEL=qwen2.5:0.5b python -m pytest tests/test_ollama_integration.py -v

OLLAMA_BASE_URL overrides the default http://localhost:11434/v1.
"""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.database.connection import get_db
from app.main import app
from app.providers.base import ChatMessage, ChatRequest, ModelNotFoundError
from app.providers.openai_compatible import OpenAICompatibleProvider

MODEL = os.environ.get("OLLAMA_INTEGRATION_MODEL")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

pytestmark = pytest.mark.skipif(
    not MODEL, reason="set OLLAMA_INTEGRATION_MODEL to run against a real Ollama server"
)


def test_provider_against_real_ollama():
    provider = OpenAICompatibleProvider(BASE_URL, timeout_seconds=300)

    async def scenario():
        try:
            health = await provider.health()
            models = await provider.list_models()
            response = await provider.chat(
                ChatRequest(
                    model=MODEL,
                    messages=(ChatMessage("user", "Reply with exactly one word: hello"),),
                    max_tokens=10,
                )
            )
            with pytest.raises(ModelNotFoundError):
                await provider.chat(
                    ChatRequest(model="model-that-does-not-exist", messages=(ChatMessage("user", "hi"),))
                )
            return health, models, response
        finally:
            await provider.aclose()

    health, models, response = asyncio.run(scenario())

    assert health.healthy
    assert MODEL in {m.id for m in models}
    assert response.message.content.strip()
    assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0
    assert response.finish_reason in ("stop", "length")


def test_chat_completions_endpoint_against_real_ollama(monkeypatch, db_session, auth_headers):
    monkeypatch.setattr(settings, "chat_provider", "openai_compatible")
    monkeypatch.setattr(settings, "openai_compatible_base_url", BASE_URL)
    monkeypatch.setattr(settings, "openai_compatible_timeout_seconds", 300)
    app.dependency_overrides[get_db] = lambda: db_session

    try:
        with TestClient(app, headers=auth_headers) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": MODEL, "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 10},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"].strip()
    assert body["usage"]["total_tokens"] > 0
