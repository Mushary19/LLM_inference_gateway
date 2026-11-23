"""OpenAICompatibleProvider tests against mocked HTTP (httpx MockTransport). No server needed."""

import asyncio
import json
from types import SimpleNamespace

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.database.connection import get_db
from app.main import app
from app.providers import openai_compatible
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    InvalidProviderRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.providers.factory import create_provider
from app.providers.fake import FakeProvider
from app.providers.registry import FakeProviderConfig, OpenAICompatibleProviderConfig
from app.providers.openai_compatible import OpenAICompatibleProvider

BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:0.5b"


def completion(content="Paris.", finish_reason="stop", model=MODEL, usage=True):
    """An Ollama-style /v1/chat/completions response body."""
    body = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "system_fingerprint": "fp_ollama",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    return body


class Upstream:
    """Mock upstream: records requests and answers with a fixed response or raises."""

    def __init__(self, response=None, raises: Exception | None = None):
        self.response = response if response is not None else httpx.Response(200, json=completion())
        self.raises = raises
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        return self.response


def make_provider(upstream: Upstream, api_key: str | None = None, name: str = "openai_compatible") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(BASE_URL, api_key=api_key, transport=httpx.MockTransport(upstream), name=name)


def chat_request(**overrides) -> ChatRequest:
    fields = {
        "model": MODEL,
        "messages": (ChatMessage("system", "Be brief."), ChatMessage("user", "Capital of France?")),
    }
    return ChatRequest(**{**fields, **overrides})


def run(coro):
    return asyncio.run(coro)


async def chat_once(provider, request):
    try:
        return await provider.chat(request)
    finally:
        await provider.aclose()


# --- Request mapping ---


def test_chat_request_is_mapped_to_openai_payload():
    upstream = Upstream()

    run(chat_once(make_provider(upstream), chat_request(max_tokens=50, temperature=0.2, top_p=0.9)))

    [sent] = upstream.requests
    assert sent.method == "POST"
    assert str(sent.url) == f"{BASE_URL}/chat/completions"
    assert json.loads(sent.content) == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Capital of France?"},
        ],
        "stream": False,
        "max_tokens": 50,
        "temperature": 0.2,
        "top_p": 0.9,
    }


def test_unset_optional_parameters_are_omitted():
    upstream = Upstream()

    run(chat_once(make_provider(upstream), chat_request()))

    payload = json.loads(upstream.requests[0].content)
    assert set(payload) == {"model", "messages", "stream"}


def test_api_key_is_sent_as_bearer_token_only_when_configured():
    with_key, without_key = Upstream(), Upstream()

    run(chat_once(make_provider(with_key, api_key="sk-test"), chat_request()))
    run(chat_once(make_provider(without_key), chat_request()))

    assert with_key.requests[0].headers["authorization"] == "Bearer sk-test"
    assert "authorization" not in without_key.requests[0].headers


# --- Response mapping ---


def test_response_is_mapped_to_normalized_chat_response():
    response = run(chat_once(make_provider(Upstream()), chat_request()))

    assert response.message == ChatMessage("assistant", "Paris.")
    assert response.model == MODEL
    assert response.finish_reason == "stop"
    assert response.usage == Usage(prompt_tokens=12, completion_tokens=3)


@pytest.mark.parametrize(
    "upstream_reason, expected", [("stop", "stop"), ("length", "length"), ("tool_calls", "stop"), (None, "stop")]
)
def test_finish_reason_is_normalized(upstream_reason, expected):
    upstream = Upstream(httpx.Response(200, json=completion(finish_reason=upstream_reason)))

    assert run(chat_once(make_provider(upstream), chat_request())).finish_reason == expected


def test_missing_usage_is_reported_as_unknown_not_zero():
    upstream = Upstream(httpx.Response(200, json=completion(usage=False)))

    assert run(chat_once(make_provider(upstream), chat_request())).usage is None


def test_missing_model_in_response_falls_back_to_requested_model():
    body = completion()
    del body["model"]

    response = run(chat_once(make_provider(Upstream(httpx.Response(200, json=body))), chat_request()))

    assert response.model == MODEL


def test_null_content_becomes_empty_string():
    upstream = Upstream(httpx.Response(200, json=completion(content=None)))

    assert run(chat_once(make_provider(upstream), chat_request())).message.content == ""


# --- Error translation ---


@pytest.mark.parametrize(
    "exception, expected",
    [
        (httpx.ConnectError("Connection refused"), ProviderUnavailableError),
        (httpx.ReadError("connection reset"), ProviderUnavailableError),
        (httpx.ConnectTimeout("connect timed out"), ProviderTimeoutError),
        (httpx.ReadTimeout("read timed out"), ProviderTimeoutError),
    ],
)
def test_transport_failures_are_normalized(exception, expected):
    with pytest.raises(expected) as excinfo:
        run(chat_once(make_provider(Upstream(raises=exception)), chat_request()))

    assert excinfo.value.__cause__ is exception


def test_ollama_missing_model_is_model_not_found():
    # What Ollama's /v1 API returns for a model that hasn't been pulled.
    body = {"error": {"message": 'model "llama9" not found, try pulling it first', "type": "api_error"}}
    upstream = Upstream(httpx.Response(404, json=body))

    with pytest.raises(ModelNotFoundError, match="llama9.*not found"):
        run(chat_once(make_provider(upstream), chat_request(model="llama9")))


def test_openai_style_model_not_found_code_is_model_not_found():
    body = {"error": {"message": "The model does not exist", "code": "model_not_found"}}

    with pytest.raises(ModelNotFoundError):
        run(chat_once(make_provider(Upstream(httpx.Response(400, json=body))), chat_request()))


@pytest.mark.parametrize(
    "status, expected",
    [
        (400, InvalidProviderRequestError),
        (413, InvalidProviderRequestError),
        (422, InvalidProviderRequestError),
        (408, ProviderTimeoutError),
        (504, ProviderTimeoutError),
        (429, ProviderUnavailableError),
        (502, ProviderUnavailableError),
        (503, ProviderUnavailableError),
        (401, ProviderError),
        (403, ProviderError),
        (500, ProviderError),
    ],
)
def test_http_errors_are_normalized(status, expected):
    upstream = Upstream(httpx.Response(status, json={"error": {"message": "upstream says no"}}))

    with pytest.raises(expected) as excinfo:
        run(chat_once(make_provider(upstream), chat_request()))

    assert type(excinfo.value) is expected  # not a more specific subclass by accident
    assert f"HTTP {status}" in str(excinfo.value) and "upstream says no" in str(excinfo.value)


def test_non_json_error_body_is_included_in_the_message():
    upstream = Upstream(httpx.Response(500, text="Internal Server Error: boom"))

    with pytest.raises(ProviderError, match="boom"):
        run(chat_once(make_provider(upstream), chat_request()))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"no": "choices"}),
        httpx.Response(200, json={"choices": [{"message": "not an object"}]}),
        httpx.Response(200, json=[1, 2, 3]),
        httpx.Response(200, json={**completion(), "usage": {"prompt_tokens": "many"}}),
    ],
)
def test_invalid_responses_raise_provider_error(response):
    with pytest.raises(ProviderError) as excinfo:
        run(chat_once(make_provider(Upstream(response)), chat_request()))

    assert type(excinfo.value) is ProviderError


# --- Client lifecycle ---


def test_one_http_client_is_reused_across_requests(monkeypatch):
    created = []
    original_init = httpx.AsyncClient.__init__

    def counting_init(self, *args, **kwargs):
        created.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(openai_compatible.httpx.AsyncClient, "__init__", counting_init)
    upstream = Upstream()
    provider = make_provider(upstream)

    async def three_requests():
        for _ in range(3):
            await provider.chat(chat_request())
        await provider.aclose()

    run(three_requests())

    assert len(created) == 1
    assert len(upstream.requests) == 3


def test_aclose_closes_the_http_client():
    provider = make_provider(Upstream())

    run(provider.aclose())

    assert provider._client.is_closed


# --- list_models, health, stream ---


def test_list_models_and_health():
    upstream = Upstream(
        httpx.Response(200, json={"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "library"}]})
    )
    provider = make_provider(upstream)

    async def scenario():
        try:
            return await provider.list_models(), await provider.health()
        finally:
            await provider.aclose()

    models, health = run(scenario())

    assert [(m.id, m.owned_by) for m in models] == [(MODEL, "library")]
    assert health.healthy
    assert all(str(r.url) == f"{BASE_URL}/models" and r.method == "GET" for r in upstream.requests)


def test_health_reports_unreachable_provider():
    provider = make_provider(Upstream(raises=httpx.ConnectError("Connection refused")))

    async def scenario():
        try:
            return await provider.health()
        finally:
            await provider.aclose()

    health = run(scenario())

    assert not health.healthy and "could not reach provider" in health.detail


# Streaming is tested in test_openai_compatible_streaming.py.


# --- Factory and application wiring ---


def test_factory_builds_each_provider_from_its_configuration():
    fake = create_provider("local-fake", FakeProviderConfig(type="fake"))
    config = OpenAICompatibleProviderConfig(type="openai_compatible", base_url=BASE_URL, timeout_seconds=5)
    real = create_provider("groq", config, api_key="k")

    assert isinstance(fake, FakeProvider) and fake.name == "local-fake"
    assert isinstance(real, OpenAICompatibleProvider) and real.name == "groq"
    assert real._client.base_url == httpx.URL(f"{BASE_URL}/")
    assert real._client.timeout == httpx.Timeout(5)
    assert real._client.headers["Authorization"] == "Bearer k"
    run(real.aclose())


def test_factory_without_api_key_sends_no_authorization():
    config = OpenAICompatibleProviderConfig(type="openai_compatible", base_url=BASE_URL)
    provider = create_provider("ollama", config)

    assert "Authorization" not in provider._client.headers
    assert provider._client.timeout == httpx.Timeout(60)  # the configuration default
    run(provider.aclose())


def test_factory_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="unknown provider type"):
        create_provider("x", SimpleNamespace(type="nope"))


def test_provider_names_default_to_the_adapter_type():
    assert OpenAICompatibleProvider(BASE_URL).name == "openai_compatible"
    assert FakeProvider().name == "fake"
    assert OpenAICompatibleProvider.name == "openai_compatible" and FakeProvider.name == "fake"


def test_provider_repr_hides_the_url_and_the_key():
    provider = OpenAICompatibleProvider("https://api.internal.example/v1", api_key="sk-secret-value", name="groq")

    assert repr(provider) == "OpenAICompatibleProvider(name='groq')"
    run(provider.aclose())


def test_app_creates_and_closes_the_registry_providers(monkeypatch, tmp_path):
    registry_file = tmp_path / "models.json"
    registry_file.write_text(json.dumps({
        "providers": {"local": {"type": "openai_compatible", "base_url": BASE_URL}},
        "models": {"qwen-small": {"provider": "local", "provider_model_id": MODEL}},
    }))
    monkeypatch.setattr(settings, "model_registry_file", registry_file)

    with TestClient(app):
        provider = app.state.model_registry.resolve("qwen-small").provider
        assert isinstance(provider, OpenAICompatibleProvider) and provider.name == "local"
        assert not provider._client.is_closed

    assert provider._client.is_closed


@pytest.fixture
def client_with_upstream(db_session, auth_headers, use_provider):
    """The real app (authenticated) with an OpenAICompatibleProvider talking to a mock upstream."""

    def build(upstream: Upstream, *, api_key: str | None = None, models=(MODEL,)) -> TestClient:
        use_provider(make_provider(upstream, api_key=api_key, name="groq"), models=models)
        app.dependency_overrides[get_db] = lambda: db_session
        return TestClient(app, headers=auth_headers)

    yield build
    app.dependency_overrides.pop(get_db, None)


URL = "/v1/chat/completions"
BODY = {"model": MODEL, "messages": [{"role": "user", "content": "Capital of France?"}], "max_tokens": 20}


def test_endpoint_returns_the_upstream_completion(client_with_upstream):
    upstream = Upstream()

    with client_with_upstream(upstream) as client:
        response = client.post(URL, json=BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion" and body["model"] == MODEL
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Paris."}
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    assert body["id"] != "chatcmpl-123"  # the gateway assigns its own ID
    assert json.loads(upstream.requests[0].content)["max_tokens"] == 20


@pytest.mark.parametrize(
    "upstream, status",
    [
        (Upstream(raises=httpx.ConnectError("Connection refused")), 503),
        (Upstream(raises=httpx.ReadTimeout("timed out")), 504),
        (Upstream(httpx.Response(404, json={"error": {"message": 'model "x" not found'}})), 502),
        (Upstream(httpx.Response(200, text="garbage")), 502),
    ],
)
def test_endpoint_maps_upstream_failures(client_with_upstream, upstream, status):
    with client_with_upstream(upstream) as client:
        response = client.post(URL, json=BODY)

    assert response.status_code == status
    assert "detail" in response.json()


# --- Whether a failure may have consumed tokens (used by quota accounting) ---


@pytest.mark.parametrize(
    "exception, expected_type, may_have_consumed",
    [
        (httpx.ConnectError("Connection refused"), ProviderUnavailableError, False),
        (httpx.ConnectTimeout("connect timed out"), ProviderTimeoutError, False),
        (httpx.PoolTimeout("no free connection"), ProviderTimeoutError, False),
        (httpx.ReadTimeout("read timed out"), ProviderTimeoutError, True),
        (httpx.WriteTimeout("write timed out"), ProviderTimeoutError, True),
        (httpx.ReadError("connection reset"), ProviderUnavailableError, True),
        (httpx.RemoteProtocolError("server disconnected"), ProviderUnavailableError, True),
    ],
)
def test_transport_failures_report_whether_tokens_may_have_been_consumed(exception, expected_type, may_have_consumed):
    with pytest.raises(expected_type) as excinfo:
        run(chat_once(make_provider(Upstream(raises=exception)), chat_request()))

    assert excinfo.value.may_have_consumed_tokens is may_have_consumed
    assert excinfo.value.usage is None


@pytest.mark.parametrize(
    "status, may_have_consumed",
    [(400, False), (401, False), (404, False), (422, False), (429, False), (408, False), (500, True), (502, True), (503, True), (504, True)],
)
def test_http_errors_report_whether_tokens_may_have_been_consumed(status, may_have_consumed):
    upstream = Upstream(httpx.Response(status, json={"error": {"message": "no"}}))

    with pytest.raises(ProviderError) as excinfo:
        run(chat_once(make_provider(upstream), chat_request()))

    assert excinfo.value.may_have_consumed_tokens is may_have_consumed


def test_unreadable_success_response_may_have_consumed_tokens():
    with pytest.raises(ProviderError) as excinfo:
        run(chat_once(make_provider(Upstream(httpx.Response(200, text="garbage"))), chat_request()))

    assert excinfo.value.may_have_consumed_tokens is True


@pytest.mark.parametrize(
    "usage",
    [{"prompt_tokens": 3}, {"prompt_tokens": -1, "completion_tokens": 2}, {"prompt_tokens": 1.5, "completion_tokens": 2}, {"prompt_tokens": True, "completion_tokens": 2}],
)
def test_malformed_usage_is_rejected_rather_than_guessed(usage):
    upstream = Upstream(httpx.Response(200, json={**completion(), "usage": usage}))

    with pytest.raises(ProviderError):
        run(chat_once(make_provider(upstream), chat_request()))


# --- Nothing upstream-specific reaches the client ---

SECRET = "sk-test-secret-0123456789"
PUBLIC = "llama-3.3-70b"
UPSTREAM_ID = "llama-3.3-70b-versatile"


def test_routed_request_uses_the_upstream_model_id_and_the_key(client_with_upstream):
    upstream = Upstream(httpx.Response(200, json=completion(model=UPSTREAM_ID)))

    with client_with_upstream(upstream, api_key=SECRET, models={PUBLIC: UPSTREAM_ID}) as client:
        response = client.post(URL, json={**BODY, "model": PUBLIC})

    assert response.status_code == 200 and response.json()["model"] == PUBLIC
    [sent] = upstream.requests
    assert json.loads(sent.content)["model"] == UPSTREAM_ID
    assert sent.headers["Authorization"] == f"Bearer {SECRET}"
    assert UPSTREAM_ID not in response.text and SECRET not in response.text


@pytest.mark.parametrize(
    "upstream, status",
    [
        (Upstream(raises=httpx.ConnectError(f"[Errno 111] connecting to {BASE_URL}")), 503),
        (Upstream(raises=httpx.ReadTimeout(f"timed out reading {BASE_URL}")), 504),
        (Upstream(httpx.Response(500, json={"error": {"message": f"crash serving {UPSTREAM_ID} on 10.0.3.7"}})), 502),
        (Upstream(httpx.Response(401, json={"error": {"message": "Invalid API key provided"}})), 502),
        (Upstream(httpx.Response(404, json={"error": {"message": f"The model {UPSTREAM_ID} does not exist"}})), 502),
        (Upstream(httpx.Response(429, json={"error": {"message": f"Rate limit reached for {UPSTREAM_ID}"}})), 503),
        (Upstream(httpx.Response(200, text=f"<html>{BASE_URL} proxy error</html>")), 502),
    ],
    ids=["connect", "timeout", "500", "401", "404", "429", "garbage"],
)
def test_upstream_failures_leak_nothing(client_with_upstream, upstream, status, caplog):
    with client_with_upstream(upstream, api_key=SECRET, models={PUBLIC: UPSTREAM_ID}) as client:
        response = client.post(URL, json={**BODY, "model": PUBLIC})

    assert response.status_code == status
    for leaked in (SECRET, UPSTREAM_ID, "localhost", "11434", "10.0.3.7", "groq", "API key", "html"):
        assert leaked not in response.text
    # Only metadata is logged: no credentials and no upstream text. (The provider model ID may
    # appear, from the registry, in the misconfiguration error for a 404.)
    for leaked in (SECRET, "10.0.3.7", "Errno 111", "timed out reading", "Invalid API key", "Rate limit reached", "proxy error", "11434"):
        assert leaked not in caplog.text


def test_upstream_400_keeps_a_useful_sanitized_message(client_with_upstream, caplog):
    message = f"Model {UPSTREAM_ID} context window is 8192 tokens; see {BASE_URL}/docs"
    upstream = Upstream(httpx.Response(400, json={"error": {"message": message}}))

    with client_with_upstream(upstream, api_key=SECRET, models={PUBLIC: UPSTREAM_ID}) as client:
        response = client.post(URL, json={**BODY, "model": PUBLIC})

    assert response.status_code == 400
    assert response.json() == {
        "detail": f"The provider rejected the request: Model {PUBLIC} context window is 8192 tokens; see [redacted]"
    }
    # The log has the failure's metadata, not the upstream message (the client has it, sanitized).
    assert "InvalidProviderRequestError, upstream status 400" in caplog.text
    assert "context window" not in caplog.text and UPSTREAM_ID not in caplog.text
    assert SECRET not in caplog.text


# --- HTTP status is kept on errors (used by provider diagnostics) ---


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 413, 418, 422, 429, 500, 502, 503, 504])
def test_http_errors_carry_the_status_code(status):
    upstream = Upstream(httpx.Response(status, json={"error": {"message": "no"}}))

    with pytest.raises(ProviderError) as excinfo:
        run(chat_once(make_provider(upstream), chat_request()))

    assert excinfo.value.status_code == status


@pytest.mark.parametrize(
    "upstream",
    [Upstream(raises=httpx.ConnectError("refused")), Upstream(raises=httpx.ReadTimeout("slow")), Upstream(httpx.Response(200, text="garbage"))],
    ids=["transport", "timeout", "unreadable"],
)
def test_non_http_errors_have_no_status_code(upstream):
    with pytest.raises(ProviderError) as excinfo:
        run(chat_once(make_provider(upstream), chat_request()))

    assert excinfo.value.status_code is None


def test_status_code_is_optional_for_existing_callers():
    error = ProviderUnavailableError("down", may_have_consumed_tokens=False)

    assert error.status_code is None and error.may_have_consumed_tokens is False
