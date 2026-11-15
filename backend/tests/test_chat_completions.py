import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api.chat import get_chat_provider
from app.core.config import settings
from app.database.connection import get_db
from app.database.models import InferenceRequest
from app.main import app
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
from app.providers.fake import FAKE_MODEL, FakeProvider
from app.services.admission import AdmissionController

URL = "/v1/chat/completions"
BODY = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "What is KV caching?"}]}


@pytest.fixture
def client(client, auth_headers):
    """Requests in this module are authenticated; authentication itself is tested in test_auth.py."""
    client.headers.update(auth_headers)
    return client


@pytest.fixture
def use_provider():
    """Inject a provider through FastAPI's dependency overrides."""

    def install(provider):
        app.dependency_overrides[get_chat_provider] = lambda: provider
        return provider

    yield install
    app.dependency_overrides.pop(get_chat_provider, None)


class RecordingProvider:
    name = "recording"

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            model="upstream-model-v2",
            message=ChatMessage("assistant", "recorded"),
            finish_reason="stop",
            usage=Usage(prompt_tokens=7, completion_tokens=1),
        )


# --- Success and response shape ---


def test_successful_chat_completion(client):
    response = client.post(URL, json=BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Fake response to: What is KV caching?",
    }
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 4, "completion_tokens": 7, "total_tokens": 11}


def test_response_has_the_normalized_openai_structure(client):
    body = client.post(URL, json=BODY).json()

    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-") and len(body["id"]) > len("chatcmpl-")
    assert isinstance(body["created"], int) and body["created"] > 0
    assert body["model"] == FAKE_MODEL
    assert len(body["choices"]) == 1
    assert set(body["choices"][0]) == {"index", "message", "finish_reason"}
    assert body["choices"][0]["index"] == 0


def test_each_response_gets_a_unique_id(client):
    ids = {client.post(URL, json=BODY).json()["id"] for _ in range(3)}

    assert len(ids) == 3


def test_max_tokens_reaches_the_provider(client):
    body = client.post(URL, json={**BODY, "max_tokens": 2}).json()

    assert body["choices"][0]["message"]["content"] == "Fake response"
    assert body["choices"][0]["finish_reason"] == "length"


def test_unknown_request_fields_are_ignored_for_sdk_compatibility(client):
    response = client.post(URL, json={**BODY, "n": 1, "user": "abc", "presence_penalty": 0})

    assert response.status_code == 200


# --- Invalid requests ---


@pytest.mark.parametrize(
    "body",
    [
        {"messages": BODY["messages"]},  # missing model
        {**BODY, "model": ""},
        {"model": FAKE_MODEL},  # missing messages
        {**BODY, "messages": []},
        {**BODY, "messages": [{"role": "tool", "content": "x"}]},
        {**BODY, "messages": [{"role": "user"}]},
        {**BODY, "messages": [{"role": "user", "content": 42}]},
        {**BODY, "max_tokens": 0},
        {**BODY, "max_tokens": -5},
        {**BODY, "temperature": 2.5},
        {**BODY, "temperature": -0.1},
        {**BODY, "top_p": 0},
        {**BODY, "top_p": 1.5},
        {**BODY, "stream": True},
    ],
)
def test_invalid_requests_are_rejected(client, body):
    assert client.post(URL, json=body).status_code == 422


def test_max_tokens_above_the_generation_limit_is_rejected(client):
    at_limit = client.post(URL, json={**BODY, "max_tokens": settings.max_generation_tokens})
    above = client.post(URL, json={**BODY, "max_tokens": settings.max_generation_tokens + 1})

    assert at_limit.status_code == 200
    assert above.status_code == 422


def test_streaming_request_gets_a_clear_message(client):
    response = client.post(URL, json={**BODY, "stream": True})

    assert "streaming is not supported yet" in response.text


# --- Provider failures ---


def test_unknown_model_returns_404(client):
    response = client.post(URL, json={**BODY, "model": "no-such-model"})

    assert response.status_code == 404
    assert "no-such-model" in response.json()["detail"]


@pytest.mark.parametrize(
    "error, expected_status",
    [
        (ProviderUnavailableError("upstream down"), 503),
        (ProviderTimeoutError("took too long"), 504),
        (ModelNotFoundError("gone"), 404),
        (InvalidProviderRequestError("context too long"), 400),
        (ProviderError("something odd"), 502),
    ],
)
def test_provider_errors_map_to_http_status(client, use_provider, error, expected_status):
    use_provider(FakeProvider(fail_with=error))

    response = client.post(URL, json=BODY)

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(error)}


def test_unexpected_provider_exception_is_a_500(use_provider, db_session, auth_headers):
    class BrokenProvider:
        name = "broken"

        async def chat(self, request):
            raise RuntimeError("bug in provider")

    use_provider(BrokenProvider())
    app.dependency_overrides[get_db] = lambda: db_session

    try:
        with TestClient(app, raise_server_exceptions=False, headers=auth_headers) as client:
            response = client.post(URL, json=BODY)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 500


# --- Dependency injection ---


def test_startup_creates_the_configured_provider(client):
    assert isinstance(app.state.chat_provider, FakeProvider)


def test_injected_provider_receives_the_normalized_request(client, use_provider):
    provider = use_provider(RecordingProvider())

    response = client.post(
        URL,
        json={
            "model": "any-model",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 10,
            "temperature": 0.3,
            "top_p": 0.9,
        },
    )

    assert response.status_code == 200
    assert provider.requests == [
        ChatRequest(
            model="any-model",
            messages=(ChatMessage("system", "Be brief."), ChatMessage("user", "Hello")),
            max_tokens=10,
            temperature=0.3,
            top_p=0.9,
        )
    ]
    body = response.json()
    assert body["model"] == "upstream-model-v2"  # the model the provider reports
    assert body["choices"][0]["message"]["content"] == "recorded"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}


def test_endpoint_bypasses_the_legacy_pipeline(client, db_session, monkeypatch):
    def admission_must_not_run(*args, **kwargs):
        raise AssertionError("chat completions must not use the admission controller")

    monkeypatch.setattr(AdmissionController, "evaluate", admission_must_not_run)

    response = client.post(URL, json=BODY)

    assert response.status_code == 200
    assert db_session.scalar(select(func.count()).select_from(InferenceRequest)) == 0
    assert app.state.scheduler is None
