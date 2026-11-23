"""POST /v1/chat/completions with "stream": true: SSE format, errors, accounting, lifecycle.

Providers here are scripted in-process streams (or the fake provider). Disconnects and
cancellation are exercised by calling the ASGI app directly with scripted receive/send, and
once through a real uvicorn server on 127.0.0.1.
"""

import asyncio
import json
import socket
import threading
import time
import uuid
from datetime import timedelta

import anyio
import psycopg
import pytest
import uvicorn
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from conftest import delete_committed_application

from app.api import chat as chat_api
from app.api.chat_stream import (
    GatewayStreamingResponse,
    StreamSettlement,
    sse_events,
    stream_meta,
)
from app.database.connection import SessionLocal, engine, get_db
from app.database.models import QuotaReservation, ReservationStatus, TokenQuota, UsageRecord, UsageStatus
from app.main import app
from app.providers.base import (
    ChatChunk,
    InvalidProviderRequestError,
    ModelInfo,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.providers.fake import FAKE_MODEL, FakeProvider
from app.providers.registry import ModelRegistry, ModelRoute
from app.schemas.chat import ChatCompletionChunk, StreamErrorEvent
from app.services import quota as quota_service
from app.services.api_keys import create_api_key, create_application
from app.services.quota import MESSAGE_OVERHEAD_TOKENS, REQUEST_OVERHEAD_TOKENS, quota_inconsistencies

URL = "/v1/chat/completions"
BODY = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": True}
ESTIMATE = len("hi") + MESSAGE_OVERHEAD_TOKENS + REQUEST_OVERHEAD_TOKENS + 10  # 92
SECRET_DETAIL = "upstream http://10.1.2.3:8000/v1 said: key sk-live-123 rejected, Traceback (most recent call last)"


@pytest.fixture
def client(client, auth_headers):
    client.headers.update(auth_headers)
    return client


# --- Helpers ---


def events(text: str) -> list:
    """The data payloads of an SSE body, JSON-decoded except [DONE]."""
    assert text.endswith("\n\n"), "every event ends with a blank line"
    payloads = []
    for block in text.split("\n\n")[:-1]:
        assert block.startswith("data: ") and "\n" not in block, block
        data = block[len("data: "):]
        payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


def content_of(payloads) -> str:
    return "".join(
        p["choices"][0]["delta"].get("content", "") for p in payloads if isinstance(p, dict) and p.get("choices")
    )


class Gate:
    """A threading.Event that async code can wait on without blocking cancellation."""

    def __init__(self):
        self._event = threading.Event()

    def set(self):
        self._event.set()

    async def wait(self, timeout=10):
        deadline = time.monotonic() + timeout
        while not self._event.is_set():
            if time.monotonic() > deadline:
                raise AssertionError("gate never opened")
            await asyncio.sleep(0.005)


HANG = "hang"


class ScriptedStream:
    """A provider whose stream follows a script.

    `before_accept` is raised before anything is yielded; then an accepted chunk (unless
    accept=False); then each step: a ChatChunk is yielded, an exception raised, a Gate waited
    on, HANG waits forever. Records calls, requests, whether the stream was closed and how.
    """

    def __init__(self, steps=(), *, before_accept=None, accept=True, accept_delay=0.0, name="scripted"):
        self.name = name
        self.steps = list(steps)
        self.before_accept = before_accept
        self.accept = accept
        self.accept_delay = accept_delay
        self.requests = []
        self.closed = 0
        self.ended_by = None
        self.accepted_at = None
        self.closed_at = None
        self.progress = []

    async def stream(self, request):
        self.requests.append(request)
        try:
            if self.accept_delay:
                await asyncio.sleep(self.accept_delay)
            if self.before_accept is not None:
                raise self.before_accept
            if self.accept:
                self.accepted_at = time.monotonic()
                yield ChatChunk("")
            for step in self.steps:
                if isinstance(step, BaseException):
                    raise step
                if isinstance(step, Gate):
                    await step.wait()
                elif step == HANG:
                    await asyncio.sleep(3600)
                elif callable(step):
                    await step()
                else:
                    self.progress.append(step)
                    yield step
            self.ended_by = "end"
        except GeneratorExit:
            self.ended_by = "closed"
            raise
        except asyncio.CancelledError:
            self.ended_by = "cancelled"
            raise
        except BaseException as exc:
            self.ended_by = type(exc).__name__
            raise
        finally:
            self.closed += 1
            self.closed_at = time.monotonic()

    async def chat(self, request):
        raise AssertionError("streaming requests must not call chat()")

    async def list_models(self):
        return [ModelInfo(id=FAKE_MODEL, owned_by=self.name)]

    async def aclose(self):
        pass


def finished(usage=Usage(3, 4), reason="stop"):
    return ChatChunk("", reason, usage)


def reservations(db, application_id):
    db.expire_all()
    return list(db.scalars(select(QuotaReservation).where(QuotaReservation.application_id == application_id)))


def usage_rows(db, application_id):
    db.expire_all()
    return list(db.scalars(select(UsageRecord).where(UsageRecord.application_id == application_id)))


def quota_of(db, application_id):
    db.expire_all()
    return db.get(TokenQuota, application_id)


def assert_settled_once(db, application_id, *, charged, status, reported, reservation_status=ReservationStatus.RECONCILED):
    [reservation] = reservations(db, application_id)
    assert (reservation.status, reservation.charged_tokens) == (reservation_status, charged)
    [record] = usage_rows(db, application_id)
    assert (record.status, record.charged_tokens, record.usage_reported) == (status, charged, reported)
    q = quota_of(db, application_id)
    assert (q.reserved_tokens, q.consumed_tokens) == (0, charged)


# --- Response format ---


def test_streaming_response_format(client):
    response = client.post(URL, json={**BODY, "max_tokens": None, "messages": [{"role": "user", "content": "What is KV caching?"}]})

    assert response.status_code == 200
    payloads = events(response.text)
    assert payloads[-1] == "[DONE]"
    chunks = payloads[:-1]
    for chunk in chunks:
        ChatCompletionChunk.model_validate(chunk)
        assert set(chunk) == {"id", "object", "created", "model", "choices"}  # no usage unless asked
        assert chunk["object"] == "chat.completion.chunk"
    assert len({c["id"] for c in chunks}) == 1 and chunks[0]["id"].startswith("chatcmpl-")
    assert len({c["created"] for c in chunks}) == 1
    assert {c["model"] for c in chunks} == {FAKE_MODEL}
    assert chunks[0]["choices"] == [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    for middle in chunks[1:-1]:
        assert middle["choices"][0]["finish_reason"] is None and set(middle["choices"][0]["delta"]) == {"content"}
    assert chunks[-1]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert content_of(chunks) == "Fake response to: What is KV caching?"


def test_the_readme_example_is_exact(client):
    response = client.post(URL, json={"model": FAKE_MODEL, "messages": [{"role": "user", "content": "Hi"}],
                                      "stream": True, "stream_options": {"include_usage": True}})

    payloads = events(response.text)
    common = {"id": payloads[0]["id"], "object": "chat.completion.chunk", "created": payloads[0]["created"], "model": FAKE_MODEL}

    def choice(delta, finish=None):
        return {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    assert payloads == [
        choice({"role": "assistant", "content": ""}),
        choice({"content": "Fake"}), choice({"content": " response"}), choice({"content": " to:"}), choice({"content": " Hi"}),
        choice({}, "stop"),
        {**common, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5}},
        "[DONE]",
    ]
    # Compact JSON, one data line per event, exactly as documented.
    assert response.text.startswith('data: {"id":"chatcmpl-')
    assert ',"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":4,"total_tokens":5}}\n\ndata: [DONE]\n\n' in response.text


def test_error_event_bytes_are_exact(client, use_provider):
    """The exact bytes checked against the openai SDK (1.90 and 3.3): a data-only event."""
    use_provider(ScriptedStream([ChatChunk("a"), ProviderTimeoutError("slow")]))

    text = client.post(URL, json=BODY).text

    assert text.endswith(
        '\n\ndata: {"error":{"message":"The provider did not respond in time.","type":"provider_error","code":"provider_timeout"}}\n\n'
    )
    assert "event:" not in text


def test_streamed_content_matches_the_non_streamed_completion(client):
    streamed = content_of(events(client.post(URL, json=BODY).text))
    plain = client.post(URL, json={**BODY, "stream": False}).json()["choices"][0]["message"]["content"]

    assert streamed == plain


def test_sse_and_rate_limit_headers(client):
    first = client.post(URL, json=BODY)
    second = client.post(URL, json=BODY)

    assert first.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert first.headers["cache-control"] == "no-cache"
    assert first.headers["x-accel-buffering"] == "no"
    assert "content-length" not in first.headers
    for name in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"):
        assert first.headers[name].isdigit(), name
    assert int(second.headers["x-ratelimit-remaining"]) == int(first.headers["x-ratelimit-remaining"]) - 1
    plain = client.post(URL, json={**BODY, "stream": False})
    assert int(plain.headers["x-ratelimit-remaining"]) == int(second.headers["x-ratelimit-remaining"]) - 1


def test_public_model_id_in_every_chunk_and_provider_model_upstream(client, use_provider):
    provider = use_provider(ScriptedStream([ChatChunk("hi"), finished()]), models={"public-name": "upstream-secret-v2"})

    response = client.post(URL, json={**BODY, "model": "public-name"})

    assert provider.requests[0].model == "upstream-secret-v2"
    assert provider.requests[0].max_tokens == 10
    assert {p["model"] for p in events(response.text)[:-1]} == {"public-name"}
    assert "upstream-secret-v2" not in response.text


def test_finish_reason_length(client, use_provider):
    use_provider(ScriptedStream([ChatChunk("a"), finished(reason="length")]))

    assert events(client.post(URL, json=BODY).text)[-2]["choices"][0]["finish_reason"] == "length"


def test_a_terminal_chunk_with_content_sends_the_content_first(client, use_provider):
    use_provider(ScriptedStream([ChatChunk("last words", "stop", Usage(1, 2))], accept=False))

    payloads = events(client.post(URL, json=BODY).text)

    assert content_of(payloads) == "last words"
    assert payloads[-2]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}
    assert payloads[-1] == "[DONE]"


# --- Usage ---


def test_usage_chunk_only_when_requested(client, use_provider):
    use_provider(ScriptedStream([ChatChunk("a"), finished(Usage(3, 4))]))

    without = events(client.post(URL, json=BODY).text)
    with_usage = events(client.post(URL, json={**BODY, "stream_options": {"include_usage": True}}).text)
    explicitly_off = events(client.post(URL, json={**BODY, "stream_options": {"include_usage": False}}).text)

    assert all("usage" not in p for p in without[:-1] + explicitly_off[:-1])
    assert with_usage[-2] == {
        "id": with_usage[0]["id"], "object": "chat.completion.chunk", "created": with_usage[0]["created"],
        "model": FAKE_MODEL, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    assert with_usage[-3]["choices"][0]["finish_reason"] == "stop"
    assert with_usage[-1] == "[DONE]"
    assert len(with_usage) == len(without) + 1


def test_unknown_usage_is_null_and_charges_the_reservation(client, use_provider, db_session, api_key):
    use_provider(ScriptedStream([ChatChunk("a"), finished(usage=None)]))

    payloads = events(client.post(URL, json={**BODY, "stream_options": {"include_usage": True}}).text)

    assert payloads[-2]["usage"] is None and payloads[-2]["choices"] == []
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.SUCCEEDED, reported=False)


def test_reported_usage_is_charged(client, use_provider, db_session, api_key):
    use_provider(ScriptedStream([ChatChunk("a"), finished(Usage(30, 5))]))

    client.post(URL, json=BODY)

    assert_settled_once(db_session, api_key.record.application_id, charged=35, status=UsageStatus.SUCCEEDED, reported=True)
    [record] = usage_rows(db_session, api_key.record.application_id)
    assert (record.prompt_tokens, record.completion_tokens, record.model, record.provider) == (30, 5, FAKE_MODEL, "scripted")


def test_usage_above_the_reservation_is_charged_in_full(client, use_provider, db_session, api_key):
    use_provider(ScriptedStream([ChatChunk("a"), finished(Usage(500, 500))]))

    client.post(URL, json=BODY)

    assert quota_of(db_session, api_key.record.application_id).consumed_tokens == 1000


def test_stream_ids_are_the_usage_request_ids(client, db_session, api_key):
    payloads = events(client.post(URL, json=BODY).text)

    [record] = usage_rows(db_session, api_key.record.application_id)
    assert payloads[0]["id"] == f"chatcmpl-{record.request_id.hex}"


# --- Errors before the stream starts: ordinary HTTP errors ---


def test_unknown_model_is_a_404_without_a_reservation(client, db_session, api_key):
    response = client.post(URL, json={**BODY, "model": "no-such-model"})

    assert response.status_code == 404
    assert response.json() == {"detail": {"code": "model_not_found", "message": "The requested model was not found."}}
    assert reservations(db_session, api_key.record.application_id) == []


def test_disabled_model_is_a_404_without_a_reservation(client, use_registry, db_session, api_key):
    provider = ScriptedStream([finished()])
    use_registry(ModelRegistry({"scripted": provider}, [ModelRoute(FAKE_MODEL, provider, FAKE_MODEL)],
                               disabled=[ModelRoute("off-model", provider, "off")]))

    response = client.post(URL, json={**BODY, "model": "off-model"})

    assert response.status_code == 404 and response.json()["detail"]["code"] == "model_not_found"
    assert provider.requests == [] and reservations(db_session, api_key.record.application_id) == []


def test_insufficient_quota_is_a_429_before_the_provider(client, use_provider, db_session, api_key):
    provider = use_provider(ScriptedStream([finished()]))

    response = client.post(URL, json={**BODY, "messages": [{"role": "user", "content": "x" * 2_000_000}]})

    assert response.status_code == 429 and response.json()["detail"]["code"] == "insufficient_quota"
    assert provider.requests == []


@pytest.mark.parametrize(
    "error, expected_status, expected_detail, charged",
    [
        (ProviderUnavailableError("down", may_have_consumed_tokens=False), 503, "The provider is temporarily unavailable.", 0),
        (ProviderTimeoutError("slow"), 504, "The provider did not respond in time.", ESTIMATE),
        (ModelNotFoundError("gone"), 502, "The provider could not serve the requested model.", 0),
        (InvalidProviderRequestError("context too long"), 400, "The provider rejected the request: context too long", 0),
        (ProviderError(SECRET_DETAIL), 502, "The provider returned an unexpected response.", ESTIMATE),
        (ProviderError("reported", usage=Usage(2, 2)), 502, "The provider returned an unexpected response.", 4),
    ],
)
def test_errors_before_acceptance_are_http_errors_settled_as_usual(
    client, use_provider, db_session, api_key, error, expected_status, expected_detail, charged
):
    provider = use_provider(ScriptedStream(before_accept=error))

    response = client.post(URL, json=BODY)

    assert response.status_code == expected_status
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": expected_detail}
    assert "10.1.2.3" not in response.text and "sk-live" not in response.text
    assert provider.closed == 1
    [reservation] = reservations(db_session, api_key.record.application_id)
    assert reservation.charged_tokens == charged
    if charged == 0:
        assert reservation.status is ReservationStatus.RELEASED
        assert usage_rows(db_session, api_key.record.application_id) == []
    else:
        assert usage_rows(db_session, api_key.record.application_id)[0].status is UsageStatus.FAILED


def test_fake_provider_errors_are_http_errors(client, use_provider):
    use_provider(FakeProvider(fail_with=ProviderUnavailableError("down")))

    response = client.post(URL, json=BODY)

    assert response.status_code == 503 and response.json() == {"detail": "The provider is temporarily unavailable."}


def test_a_stream_that_yields_nothing_is_a_502_not_a_success(client, use_provider, db_session, api_key, caplog):
    provider = use_provider(ScriptedStream(accept=False))

    response = client.post(URL, json=BODY)

    assert response.status_code == 502
    assert response.json() == {"detail": "The provider returned an unexpected response."}
    assert "ended a stream without any chunk" in caplog.text
    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_a_provider_bug_before_acceptance_is_a_500_and_still_settles(use_provider, db_session, api_key, auth_headers):
    provider = use_provider(ScriptedStream(before_accept=RuntimeError("bug in provider")))
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app, raise_server_exceptions=False, headers=auth_headers) as client:
            response = client.post(URL, json=BODY)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 500 and "bug in provider" not in response.text
    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


# --- Errors after the stream started: one error event, no [DONE] ---


@pytest.mark.parametrize(
    "error, code, message",
    [
        (ProviderTimeoutError(SECRET_DETAIL), "provider_timeout", "The provider did not respond in time."),
        (ProviderUnavailableError(SECRET_DETAIL), "provider_unavailable", "The provider is temporarily unavailable."),
        (ProviderError(SECRET_DETAIL), "provider_error", "The provider returned an unexpected response."),
        (InvalidProviderRequestError(SECRET_DETAIL), "provider_error", "The provider returned an unexpected response."),
        (ModelNotFoundError(SECRET_DETAIL), "provider_error", "The provider returned an unexpected response."),
    ],
)
def test_mid_stream_provider_errors_are_safe_error_events(client, use_provider, db_session, api_key, error, code, message, caplog):
    provider = use_provider(ScriptedStream([ChatChunk("partial"), error]))

    response = client.post(URL, json=BODY)

    assert response.status_code == 200
    payloads = events(response.text)
    assert content_of(payloads) == "partial"
    assert payloads[-1] == {"error": {"message": message, "type": "provider_error", "code": code}}
    StreamErrorEvent.model_validate(payloads[-1])
    assert "[DONE]" not in payloads
    assert all(p["choices"][0]["finish_reason"] is None for p in payloads[:-1])  # never a finish chunk
    for leaked in ("10.1.2.3", "sk-live", "Traceback", "upstream http"):
        assert leaked not in response.text
    # Logged as metadata (error type), never with the upstream text.
    assert type(error).__name__ in caplog.text or "check the model registry" in caplog.text
    assert SECRET_DETAIL not in caplog.text
    for leaked in ("10.1.2.3", "sk-live", "upstream http"):
        assert leaked not in caplog.text
    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_error_messages_match_the_non_streaming_ones():
    from app.api.chat_stream import _DEFAULT_PROVIDER_ERROR_EVENT, _PROVIDER_ERROR_EVENTS

    http = {error_type: text for error_type, _, text in chat_api._ERROR_RESPONSES}
    for error_type, _, text in _PROVIDER_ERROR_EVENTS:
        assert http[error_type] == text
    assert _DEFAULT_PROVIDER_ERROR_EVENT[1] == chat_api._DEFAULT_ERROR_RESPONSE[1]


def test_an_incomplete_stream_is_an_error_not_a_success(client, use_provider, db_session, api_key):
    provider = use_provider(ScriptedStream([ChatChunk("cut"), ChatChunk(" off")]))  # no terminal chunk

    payloads = events(client.post(URL, json={**BODY, "stream_options": {"include_usage": True}}).text)

    assert payloads[-1] == {"error": {"message": "The provider returned an unexpected response.",
                                      "type": "provider_error", "code": "provider_error"}}
    assert "[DONE]" not in payloads and all("usage" not in p for p in payloads)
    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_a_gateway_bug_mid_stream_is_an_internal_error_event(client, use_provider, db_session, api_key, caplog):
    provider = use_provider(ScriptedStream([ChatChunk("a"), RuntimeError("bug /srv/secret.py")]))

    response = client.post(URL, json=BODY)

    payloads = events(response.text)
    assert payloads[-1] == {"error": {"message": "An internal error occurred.", "type": "server_error", "code": "internal_error"}}
    assert "secret.py" not in response.text and "Streaming failed" in caplog.text
    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


# --- Settlement ---


def test_settlement_happens_once_per_stream(client, use_provider, db_session, api_key, monkeypatch):
    calls = []
    real_settle = quota_service.settle
    monkeypatch.setattr(quota_service, "settle", lambda *a, **k: calls.append(k) or real_settle(*a, **k))
    use_provider(ScriptedStream([ChatChunk("a"), finished()]))

    for _ in range(3):
        client.post(URL, json=BODY)

    assert len(calls) == 3
    assert len(usage_rows(db_session, api_key.record.application_id)) == 3


def test_a_database_failure_during_settlement_still_completes_the_stream(client, use_provider, db_session, api_key, monkeypatch, caplog):
    def failing_settle(*args, **kwargs):
        raise OperationalError("UPDATE", {}, Exception("connection lost"))

    monkeypatch.setattr(quota_service, "settle", failing_settle)
    use_provider(ScriptedStream([ChatChunk("a"), finished()]))

    payloads = events(client.post(URL, json=BODY).text)

    assert payloads[-1] == "[DONE]"  # the client has its whole output
    assert "stays reserved until expired" in caplog.text
    [reservation] = reservations(db_session, api_key.record.application_id)
    assert reservation.status is ReservationStatus.RESERVED  # left for expire-reservations
    assert quota_of(db_session, api_key.record.application_id).reserved_tokens == ESTIMATE


def test_an_unexpected_settlement_failure_is_logged_and_does_not_break_the_stream(client, use_provider, monkeypatch, caplog):
    def broken_settle(*args, **kwargs):
        raise RuntimeError("settlement bug")

    monkeypatch.setattr(quota_service, "settle", broken_settle)
    use_provider(ScriptedStream([ChatChunk("a"), finished()]))

    assert events(client.post(URL, json=BODY).text)[-1] == "[DONE]"
    assert "Could not settle the reservation" in caplog.text


def test_settlement_precedes_done_and_the_session_teardown(use_provider, db_session, auth_headers, monkeypatch):
    order = []
    real_settle = quota_service.settle

    def recording_settle(*args, **kwargs):
        order.append("settle")
        return real_settle(*args, **kwargs)

    def tracked_db():
        try:
            yield db_session
        finally:
            order.append("teardown")

    monkeypatch.setattr(quota_service, "settle", recording_settle)
    use_provider(ScriptedStream([ChatChunk("a"), finished()]))
    app.dependency_overrides[get_db] = tracked_db
    try:
        with TestClient(app, headers=auth_headers) as client:
            sent = run_asgi(client, BODY, auth_headers, on_send=lambda m: order.append(body_kind(m)))
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert sent.done
    assert order.index("settle") < order.index("done") < order.index("teardown")
    assert order.count("settle") == 1


# --- The database connection is not held while streaming ---


@pytest.fixture
def committed_key():
    name = f"stream-{uuid.uuid4().hex[:8]}"
    with SessionLocal() as db:
        application = create_application(db, name, token_limit=ESTIMATE * 3)
        _, plaintext = create_api_key(db, application, "stream")
        application_id = application.id
        db.commit()
    yield plaintext, application_id
    with SessionLocal() as db:
        delete_committed_application(db, name)


def _idle_in_transaction_backends() -> int:
    url = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(url) as connection:
        return connection.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
            "AND state LIKE 'idle in transaction%' AND pid <> pg_backend_pid()"
        ).fetchone()[0]


def test_no_database_connection_is_held_while_streaming(committed_key, use_provider):
    plaintext, app_id = committed_key
    observations = []

    async def observe():
        observations.append((engine.pool.checkedout(), await asyncio.to_thread(_idle_in_transaction_backends)))

    use_provider(ScriptedStream([observe, ChatChunk("a"), observe, ChatChunk("b"), observe, finished()]))
    with TestClient(app) as client:
        response = client.post(URL, json=BODY, headers={"Authorization": f"Bearer {plaintext}"})
        assert engine.pool.checkedout() == 0

    assert events(response.text)[-1] == "[DONE]"
    assert observations == [(0, 0)] * 3
    with SessionLocal() as db:
        assert quota_of(db, app_id).consumed_tokens == 7


# --- Reservation expiry and concurrency ---


def test_expiry_racing_a_stream_never_charges_twice(committed_key, use_provider):
    plaintext, app_id = committed_key
    mid_stream, release = Gate(), Gate()

    async def reached():
        mid_stream.set()

    provider = use_provider(ScriptedStream([ChatChunk("a"), reached, release, ChatChunk("b"), finished(Usage(3, 4))]))
    result = {}

    with TestClient(app) as client:
        thread = threading.Thread(target=lambda: result.update(
            response=client.post(URL, json=BODY, headers={"Authorization": f"Bearer {plaintext}"})))
        thread.start()
        asyncio.run(mid_stream.wait())
        with SessionLocal() as db:
            expired = quota_service.expire_stale_reservations(db, timedelta(0))  # expires the in-flight stream
        release.set()
        thread.join(10)

    assert len(expired) == 1
    assert events(result["response"].text)[-1] == "[DONE]"  # the stream's own settlement was a no-op
    assert provider.closed == 1
    with SessionLocal() as db:
        [reservation] = reservations(db, app_id)
        assert (reservation.status, reservation.charged_tokens) == (ReservationStatus.EXPIRED, ESTIMATE)
        [record] = usage_rows(db, app_id)
        assert record.status is UsageStatus.EXPIRED
        q = quota_of(db, app_id)
        assert (q.reserved_tokens, q.consumed_tokens) == (0, ESTIMATE)
        assert quota_inconsistencies(db, app_id) == []


def test_concurrent_streams_cannot_exceed_the_quota(committed_key, use_provider):
    plaintext, app_id = committed_key
    gate = Gate()
    provider = use_provider(ScriptedStream([ChatChunk("a"), gate, ChatChunk("b"), finished(Usage(2, 3))]))
    headers = {"Authorization": f"Bearer {plaintext}"}
    responses, lock = [], threading.Lock()

    with TestClient(app) as client:
        barrier = threading.Barrier(10)

        def call():
            barrier.wait()
            response = client.post(URL, json=BODY, headers=headers)
            with lock:
                responses.append(response)

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 10
        while not (len(provider.requests) == 3 and len(responses) == 7) and time.monotonic() < deadline:
            time.sleep(0.01)
        gate.set()
        for t in threads:
            t.join(10)

    assert sorted(r.status_code for r in responses) == [200] * 3 + [429] * 7
    assert all(events(r.text)[-1] == "[DONE]" for r in responses if r.status_code == 200)
    assert len(provider.requests) == 3 and provider.closed == 3
    with SessionLocal() as db:
        q = quota_of(db, app_id)
        assert (q.consumed_tokens, q.reserved_tokens) == (15, 0)
        assert len(usage_rows(db, app_id)) == 3
        assert quota_inconsistencies(db, app_id) == []


# --- Disconnects and cancellation (direct ASGI calls) ---


class Sent:
    def __init__(self):
        self.messages = []

    @property
    def body(self) -> bytes:
        return b"".join(m.get("body", b"") for m in self.messages if m["type"] == "http.response.body")

    @property
    def done(self) -> bool:
        return self.body.endswith(b"data: [DONE]\n\n")

    @property
    def status(self):
        return next(m["status"] for m in self.messages if m["type"] == "http.response.start")


def body_kind(message) -> str:
    if message["type"] == "http.response.start":
        return "start"
    return "done" if message.get("body", b"").endswith(b"data: [DONE]\n\n") else "body"


def asgi_scope(body: bytes, headers: dict, spec_version: str) -> dict:
    all_headers = {**headers, "content-type": "application/json", "content-length": str(len(body))}
    return {
        "type": "http", "asgi": {"version": "3.0", "spec_version": spec_version}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": URL, "raw_path": URL.encode(), "root_path": "",
        "query_string": b"", "headers": [(k.lower().encode(), v.encode()) for k, v in all_headers.items()],
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80), "state": {},
    }


def run_asgi(client, body, headers, *, disconnect: Gate | None = None, on_send=None, send_hook=None,
             spec_version="2.3", disconnect_immediately=False) -> Sent:
    """Call the app as uvicorn's h11 server would (ASGI spec 2.3 by default), inside the
    TestClient's event loop. After the request body, receive() blocks until `disconnect`
    opens (or forever), then reports http.disconnect, every time it is called again."""
    raw = json.dumps(body).encode()
    sent = Sent()
    state = {"body_sent": False}

    async def receive():
        if not state["body_sent"]:
            state["body_sent"] = True
            return {"type": "http.request", "body": raw, "more_body": False}
        if not disconnect_immediately:
            if disconnect is None:
                await asyncio.sleep(3600)
            else:
                await disconnect.wait(timeout=30)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.messages.append(message)
        if on_send is not None:
            on_send(message)
        if send_hook is not None:
            await send_hook(message)

    async def scenario():
        with anyio.fail_after(20):
            await app(asgi_scope(raw, headers, spec_version), receive, send)

    client.portal.call(scenario)
    return sent


def test_client_disconnect_mid_stream_closes_the_provider_and_settles(client, use_provider, db_session, api_key, auth_headers):
    disconnect = Gate()
    provider = use_provider(ScriptedStream([ChatChunk("a"), HANG]))

    def on_send(message):
        if b'"content":"a"' in message.get("body", b""):
            disconnect.set()

    sent = run_asgi(client, BODY, auth_headers, disconnect=disconnect, on_send=on_send)

    assert sent.status == 200 and not sent.done
    assert provider.closed == 1 and provider.ended_by in ("cancelled", "closed")
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_disconnect_while_blocked_in_send_still_closes_and_settles(client, use_provider, db_session, api_key, auth_headers):
    """The disconnect cancels the streaming task while it is inside ASGI send(), so the body
    generator is suspended at a yield: Starlette alone would never close it."""
    disconnect = Gate()
    provider = use_provider(ScriptedStream([ChatChunk("a"), ChatChunk("b"), finished()]))

    async def slow_send(message):
        if b'"content":"a"' in message.get("body", b""):
            disconnect.set()
            await asyncio.sleep(3600)  # cancelled by Starlette's disconnect listener

    sent = run_asgi(client, BODY, auth_headers, disconnect=disconnect, send_hook=slow_send)

    assert not sent.done
    assert provider.closed == 1 and provider.ended_by == "closed"
    assert provider.progress == [ChatChunk("a")]  # nothing was generated after the client left
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_a_failing_send_closes_and_settles(client, use_provider, db_session, api_key, auth_headers):
    """ASGI spec 2.4: a disconnect shows up as an OSError from send()."""
    provider = use_provider(ScriptedStream([ChatChunk("a"), ChatChunk("b"), finished()]))

    async def failing_send(message):
        if b'"content":"a"' in message.get("body", b""):
            raise OSError("connection reset")

    with pytest.raises(ClientDisconnect):  # Starlette's translation of the OSError
        run_asgi(client, BODY, auth_headers, send_hook=failing_send, spec_version="2.4")

    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_cancelling_the_whole_request_mid_stream_closes_and_settles(client, use_provider, db_session, api_key, auth_headers):
    provider = use_provider(ScriptedStream([ChatChunk("a"), HANG]))
    raw = json.dumps(BODY).encode()
    received = {"n": 0}

    async def receive():
        received["n"] += 1
        if received["n"] == 1:
            return {"type": "http.request", "body": raw, "more_body": False}
        await asyncio.sleep(3600)

    async def scenario():
        sent = []

        async def send(message):
            sent.append(message)

        with anyio.move_on_after(0.5):  # e.g. server shutdown cancelling the request task
            await app(asgi_scope(raw, auth_headers, "2.3"), receive, send)
        return sent

    sent = client.portal.call(scenario)

    assert any(b'"content":"a"' in m.get("body", b"") for m in sent)
    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_disconnect_while_waiting_for_acceptance_is_noticed_once_the_stream_starts(
    client, use_provider, db_session, api_key, auth_headers
):
    """Documented limitation: before the provider accepts the request there is no response yet,
    so nothing watches the client. The wait runs to completion; the disconnect is noticed as
    soon as the response starts, and the stream is closed and settled then."""
    provider = use_provider(ScriptedStream([ChatChunk("a"), HANG], accept_delay=0.3))

    started = time.monotonic()
    sent = run_asgi(client, BODY, auth_headers, disconnect_immediately=True)

    assert provider.accepted_at is not None and provider.accepted_at - started >= 0.3  # the wait was not cut short
    assert provider.closed == 1 and provider.closed_at - provider.accepted_at < 0.5  # closed right after
    assert not sent.done
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_successful_stream_over_asgi_settles_once(client, use_provider, db_session, api_key, auth_headers):
    provider = use_provider(ScriptedStream([ChatChunk("a"), finished(Usage(1, 1))]))

    sent = run_asgi(client, BODY, auth_headers)

    # The gateway stops at the terminal chunk; the owner then closes the suspended generator.
    assert sent.done and provider.closed == 1 and provider.ended_by == "closed"
    assert_settled_once(db_session, api_key.record.application_id, charged=2, status=UsageStatus.SUCCEEDED, reported=True)


def test_client_disconnect_through_a_real_uvicorn_server(committed_key, use_provider):
    """End to end with uvicorn's h11 implementation: the client closes its socket mid-stream."""
    plaintext, app_id = committed_key
    provider = use_provider(ScriptedStream([ChatChunk("first"), HANG]))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, lifespan="on", log_level="warning", http="h11"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        body = json.dumps(BODY).encode()
        with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
            conn.sendall(
                f"POST {URL} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {plaintext}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
            )
            received = b""
            while b'"content":"first"' not in received:
                part = conn.recv(65536)
                assert part, "server closed the connection early"
                received += part
        assert b"200 OK" in received and b"text/event-stream" in received
        deadline = time.monotonic() + 5
        while provider.closed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert provider.closed == 1 and provider.ended_by in ("cancelled", "closed")
        # Settlement follows the close; give it a moment to commit.
        while time.monotonic() < deadline:
            with SessionLocal() as db:
                if reservations(db, app_id)[0].status is not ReservationStatus.RESERVED:
                    break
            time.sleep(0.02)
    finally:
        server.should_exit = True
        thread.join(10)
    with SessionLocal() as db:
        [reservation] = reservations(db, app_id)
        assert (reservation.status, reservation.charged_tokens) == (ReservationStatus.RECONCILED, ESTIMATE)
        assert usage_rows(db, app_id)[0].status is UsageStatus.FAILED


# --- StreamSettlement and GatewayStreamingResponse directly ---


class RecordingSettle:
    def __init__(self, fail: Exception | None = None):
        self.calls = []
        self.fail = fail

    async def __call__(self, **kwargs):
        await asyncio.sleep(0)  # a checkpoint, where cancellation would strike
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail


def test_finish_closes_and_settles_exactly_once():
    provider = ScriptedStream([ChatChunk("a"), HANG])
    settle = RecordingSettle()

    async def scenario():
        stream = provider.stream(None)
        await stream.__anext__()
        owner = StreamSettlement(stream, settle, uuid.uuid4())
        await owner.finish(usage=None, succeeded=False, may_have_consumed_tokens=True, outcome="client_disconnected")
        await owner.finish(usage=Usage(1, 1), succeeded=True, may_have_consumed_tokens=True, outcome="completed")
        return owner

    owner = asyncio.run(scenario())
    assert owner.finished and provider.closed == 1
    assert settle.calls == [{"usage": None, "succeeded": False, "may_have_consumed_tokens": True}]


def test_finish_is_shielded_from_cancellation():
    provider = ScriptedStream([ChatChunk("a"), HANG])
    settle = RecordingSettle()

    async def scenario():
        stream = provider.stream(None)
        await stream.__anext__()
        owner = StreamSettlement(stream, settle, uuid.uuid4())
        with anyio.CancelScope() as scope:
            scope.cancel()  # already cancelled when finish() starts
            await owner.finish(usage=None, succeeded=False, may_have_consumed_tokens=True, outcome="cancelled")
            await anyio.sleep(0)  # the cancellation was only postponed, and strikes here
        assert scope.cancelled_caught

    asyncio.run(scenario())
    assert provider.closed == 1 and len(settle.calls) == 1


def test_a_failing_close_still_settles_and_a_failing_settle_is_logged(caplog):
    class Unclosable:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("close failed")

    settle = RecordingSettle(fail=RuntimeError("settle failed"))

    asyncio.run(StreamSettlement(Unclosable(), settle, uuid.uuid4()).finish(
        usage=None, succeeded=False, may_have_consumed_tokens=True, outcome="gateway_error"))

    assert len(settle.calls) == 1
    assert "Could not close the provider stream" in caplog.text and "Could not settle" in caplog.text


def test_real_settlement_runs_even_if_cancelled_while_waiting_for_a_worker_thread(monkeypatch, db_session, api_key):
    """_settle runs quota.settle in a worker thread. anyio only shields the thread once it is
    dispatched; a cancellation while waiting for a free thread would skip it. StreamSettlement's
    own shield covers that wait."""
    held = quota_service.reserve(
        db_session, application_id=api_key.record.application_id, api_key_id=api_key.record.id,
        request_id=uuid.uuid4(), tokens=ESTIMATE, provider="scripted", model=FAKE_MODEL,
    )
    release = threading.Event()

    async def scenario():
        limiter = anyio.to_thread.current_default_thread_limiter()
        original = limiter.total_tokens
        limiter.total_tokens = 1
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(anyio.to_thread.run_sync, release.wait)  # occupies the only thread
                await anyio.sleep(0.05)
                provider = ScriptedStream([HANG])
                stream = provider.stream(None)
                await stream.__anext__()
                owner = StreamSettlement(
                    stream, lambda **kw: chat_api._settle(db_session, held, model=FAKE_MODEL, started=time.monotonic(), **kw),
                    held.request_id,
                )
                with anyio.CancelScope() as scope:
                    scope.cancel()
                    threading.Timer(0.2, release.set).start()
                    await owner.finish(usage=None, succeeded=False, may_have_consumed_tokens=True, outcome="cancelled")
                return provider
        finally:
            limiter.total_tokens = original

    provider = asyncio.run(scenario())
    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_a_response_that_never_starts_its_body_still_closes_and_settles():
    provider = ScriptedStream([ChatChunk("a"), HANG])
    settle = RecordingSettle()
    route = ModelRoute(FAKE_MODEL, provider, FAKE_MODEL)

    async def scenario():
        stream = provider.stream(None)
        first = await stream.__anext__()
        owner = StreamSettlement(stream, settle, uuid.uuid4())
        meta = stream_meta(uuid.uuid4(), FAKE_MODEL, include_usage=False)
        response = GatewayStreamingResponse(sse_events(first, stream, owner, meta, route), owner=owner, headers={})

        async def receive():
            await asyncio.sleep(3600)

        async def send(message):
            raise OSError("connection gone before the headers")

        with pytest.raises(ClientDisconnect):  # Starlette's translation of the OSError
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    asyncio.run(scenario())
    assert provider.closed == 1
    assert settle.calls == [{"usage": None, "succeeded": False, "may_have_consumed_tokens": True}]
