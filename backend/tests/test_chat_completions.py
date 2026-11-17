import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api.chat import MODEL_NOT_FOUND
from app.core.config import settings
from app.database.connection import get_db
from app.database.models import InferenceRequest, QuotaReservation, TokenQuota, UsageRecord
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
from app.providers.registry import ModelRegistry, ModelRoute
from app.services.admission import AdmissionController

URL = "/v1/chat/completions"
BODY = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "What is KV caching?"}]}


@pytest.fixture
def client(client, auth_headers):
    """Requests in this module are authenticated; authentication itself is tested in test_auth.py."""
    client.headers.update(auth_headers)
    return client


class RecordingProvider:
    def __init__(self, name: str = "recording", reply: str = "recorded") -> None:
        self.name = name
        self.reply = reply
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            model="upstream-model-v2",
            message=ChatMessage("assistant", self.reply),
            finish_reason="stop",
            usage=Usage(prompt_tokens=7, completion_tokens=1),
        )

    async def aclose(self) -> None:
        pass


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
    assert response.json() == {
        "detail": {"code": "model_not_found", "message": "The requested model was not found."}
    }


@pytest.mark.parametrize(
    "error, expected_status, expected_detail",
    [
        (ProviderUnavailableError("upstream down"), 503, "The provider is temporarily unavailable."),
        (ProviderTimeoutError("took too long"), 504, "The provider did not respond in time."),
        # The registry routed the model, so the provider not knowing it is a gateway problem.
        (ModelNotFoundError("gone"), 502, "The provider could not serve the requested model."),
        (InvalidProviderRequestError("context too long"), 400, "The provider rejected the request: context too long"),
        (InvalidProviderRequestError(""), 400, "The provider rejected the request as invalid."),
        (ProviderError("something odd"), 502, "The provider returned an unexpected response."),
    ],
)
def test_provider_errors_map_to_http_status(client, use_provider, error, expected_status, expected_detail, caplog):
    use_provider(FakeProvider(fail_with=error))

    response = client.post(URL, json=BODY)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert "Provider fake" in caplog.text and str(error) in caplog.text  # details stay in the server log


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


def test_startup_creates_the_default_registry(client):
    route = app.state.model_registry.resolve(FAKE_MODEL)

    assert isinstance(route.provider, FakeProvider)
    assert (route.provider.name, route.provider_model_id) == ("fake", FAKE_MODEL)


def test_injected_provider_receives_the_normalized_request(client, use_provider):
    provider = use_provider(RecordingProvider(), models={"any-model": "upstream-any-model"})

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
            model="upstream-any-model",  # the provider gets its own model ID
            messages=(ChatMessage("system", "Be brief."), ChatMessage("user", "Hello")),
            max_tokens=10,
            temperature=0.3,
            top_p=0.9,
        )
    ]
    body = response.json()
    assert body["model"] == "any-model"  # the public ID, not what the provider reports
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


# --- Model routing ---


def two_provider_registry():
    groq = RecordingProvider("groq", reply="from groq")
    ollama = RecordingProvider("ollama", reply="from ollama")
    registry = ModelRegistry(
        {"groq": groq, "ollama": ollama},
        [
            ModelRoute("llama-3.3-70b", groq, "llama-3.3-70b-versatile"),
            ModelRoute("llama-3.1-8b", groq, "llama-3.1-8b-instant"),
            ModelRoute("qwen-local", ollama, "qwen2.5:7b"),
        ],
        disabled=["retired-model"],
    )
    return registry, groq, ollama


@pytest.mark.parametrize(
    "public_id, provider_name, provider_model_id, reply",
    [
        ("llama-3.3-70b", "groq", "llama-3.3-70b-versatile", "from groq"),
        ("llama-3.1-8b", "groq", "llama-3.1-8b-instant", "from groq"),
        ("qwen-local", "ollama", "qwen2.5:7b", "from ollama"),
    ],
)
def test_public_model_routes_to_its_provider_and_upstream_model(
    client, use_registry, public_id, provider_name, provider_model_id, reply
):
    registry, groq, ollama = two_provider_registry()
    use_registry(registry)
    providers = {"groq": groq, "ollama": ollama}

    response = client.post(URL, json={**BODY, "model": public_id})

    assert response.status_code == 200
    assert response.json()["model"] == public_id
    assert response.json()["choices"][0]["message"]["content"] == reply
    chosen = providers[provider_name]
    [sent] = chosen.requests
    assert sent.model == provider_model_id
    other = next(p for name, p in providers.items() if name != provider_name)
    assert other.requests == []


def _quota_state(db_session, application_id):
    """(reserved, consumed, reservation count, usage record count) for one application."""
    db_session.expire_all()
    quota = db_session.get(TokenQuota, application_id)
    return (
        quota.reserved_tokens,
        quota.consumed_tokens,
        db_session.scalar(
            select(func.count()).select_from(QuotaReservation).where(QuotaReservation.application_id == application_id)
        ),
        db_session.scalar(
            select(func.count()).select_from(UsageRecord).where(UsageRecord.application_id == application_id)
        ),
    )


@pytest.mark.parametrize(
    "model",
    ["no-such-model", "retired-model", "LLAMA-3.3-70B", "llama-3.3-70b ", "llama-3.3-70b-versatile"],
    ids=["unknown", "disabled", "wrong-case", "trailing-space", "provider-model-id"],
)
def test_unknown_or_disabled_model_is_rejected_before_any_quota_work(client, use_registry, db_session, api_key, model):
    registry, groq, ollama = two_provider_registry()
    use_registry(registry)
    before = _quota_state(db_session, api_key.record.application_id)

    response = client.post(URL, json={**BODY, "model": model})

    assert response.status_code == 404
    assert response.json() == {"detail": MODEL_NOT_FOUND}
    assert groq.requests == [] and ollama.requests == []
    assert _quota_state(db_session, api_key.record.application_id) == before


def test_upstream_model_not_found_is_a_502_and_releases_the_reservation(client, use_provider, db_session, api_key):
    use_provider(FakeProvider(), models={"public-name": "missing-upstream"})  # fake only serves fake-model

    response = client.post(URL, json={**BODY, "model": "public-name"})

    assert response.status_code == 502
    assert "missing-upstream" not in response.text
    assert _quota_state(db_session, api_key.record.application_id) == (0, 0, 1, 0)
    status = db_session.scalar(
        select(QuotaReservation.status).where(QuotaReservation.application_id == api_key.record.application_id)
    )
    assert status.value == "released"  # ModelNotFoundError never consumed tokens


@pytest.mark.parametrize(
    "error",
    [
        ProviderUnavailableError("could not reach provider: http://10.0.0.5:8000/v1 refused (model llama-3.3-70b-versatile)"),
        ProviderTimeoutError("provider did not respond in time (ReadTimeout) https://internal.example/v1"),
        ProviderError("provider returned HTTP 500: upstream stack trace at /srv/app.py, model llama-3.3-70b-versatile"),
        ModelNotFoundError("provider returned HTTP 404: model 'llama-3.3-70b-versatile' does not exist"),
    ],
)
def test_server_side_provider_errors_get_generic_messages(client, use_provider, error):
    use_provider(FakeProvider(fail_with=error), models={"llama-3.3-70b": "llama-3.3-70b-versatile"})

    response = client.post(URL, json={**BODY, "model": "llama-3.3-70b"})

    assert response.status_code >= 500
    for leaked in ("10.0.0.5", "internal.example", "versatile", "/srv/app.py", "stack trace", "http"):
        assert leaked not in response.text


def test_upstream_400_messages_are_sanitized(client, use_provider):
    message = (
        "provider returned HTTP 400: model llama-3.3-70b-versatile: context length 9000 exceeds 8192;"
        " see https://docs.internal.example/limits or 192.168.1.20:8080\n\x07" + "x" * 500
    )
    use_provider(
        FakeProvider(fail_with=InvalidProviderRequestError(message)),
        models={"llama-3.3-70b": "llama-3.3-70b-versatile"},
    )

    response = client.post(URL, json={**BODY, "model": "llama-3.3-70b"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail.startswith("The provider rejected the request: model llama-3.3-70b: context length 9000 exceeds 8192;")
    for leaked in ("HTTP 400", "versatile", "docs.internal.example", "192.168.1.20", "\x07", "\n"):
        assert leaked not in detail
    assert "[redacted]" in detail
    assert len(detail) <= len("The provider rejected the request: ") + 300


def test_listing_models_returns_only_enabled_public_ids(client, use_registry):
    registry, groq, ollama = two_provider_registry()
    use_registry(registry)

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "inferenceflow"}
            for model in ("llama-3.1-8b", "llama-3.3-70b", "qwen-local")  # sorted; retired-model is disabled
        ],
    }
    for hidden in ("groq", "ollama", "versatile", "instant", "qwen2.5", "retired-model"):
        assert hidden not in response.text


def test_listing_models_never_calls_a_provider(client, use_provider):
    class UnreachableProvider:
        name = "unreachable"

        async def list_models(self):
            raise AssertionError("GET /v1/models must not ask providers")

        async def health(self):
            raise AssertionError("GET /v1/models must not ask providers")

        async def chat(self, request):
            raise ProviderUnavailableError("down")

    use_provider(UnreachableProvider())

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert [m["id"] for m in response.json()["data"]] == [FAKE_MODEL]
