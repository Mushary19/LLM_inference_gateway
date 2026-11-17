"""OpenAICompatibleProvider tests against mocked HTTP (httpx MockTransport). No server needed."""

import asyncio
import json

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_chat_provider
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


def make_provider(upstream: Upstream, api_key: str | None = None) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(BASE_URL, api_key=api_key, transport=httpx.MockTransport(upstream))


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


def test_stream_is_not_implemented_yet():
    provider = make_provider(Upstream())

    async def scenario():
        try:
            return [chunk async for chunk in provider.stream(chat_request())]
        finally:
            await provider.aclose()

    with pytest.raises(NotImplementedError):
        run(scenario())


# --- Factory and application wiring ---


def test_factory_builds_each_provider():
    fake = create_provider("fake")
    real = create_provider("openai_compatible", base_url=BASE_URL, api_key="k", timeout_seconds=5)

    assert isinstance(fake, FakeProvider)
    assert isinstance(real, OpenAICompatibleProvider)
    assert real._client.base_url == httpx.URL(f"{BASE_URL}/")
    assert real._client.timeout == httpx.Timeout(5)
    run(real.aclose())


@pytest.mark.parametrize("name, kwargs", [("openai_compatible", {}), ("nope", {"base_url": BASE_URL})])
def test_factory_rejects_bad_configuration(name, kwargs):
    with pytest.raises(ValueError):
        create_provider(name, **kwargs)


def test_app_creates_and_closes_the_configured_provider(monkeypatch):
    monkeypatch.setattr(settings, "chat_provider", "openai_compatible")

    with TestClient(app):
        provider = app.state.chat_provider
        assert isinstance(provider, OpenAICompatibleProvider)
        assert not provider._client.is_closed

    assert provider._client.is_closed


@pytest.fixture
def client_with_upstream(db_session, auth_headers):
    """The real app (authenticated) with an OpenAICompatibleProvider talking to a mock upstream."""

    def build(upstream: Upstream) -> TestClient:
        app.dependency_overrides[get_chat_provider] = lambda: make_provider(upstream)
        app.dependency_overrides[get_db] = lambda: db_session
        return TestClient(app, headers=auth_headers)

    yield build
    app.dependency_overrides.pop(get_chat_provider, None)
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
        (Upstream(httpx.Response(404, json={"error": {"message": 'model "x" not found'}})), 404),
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
