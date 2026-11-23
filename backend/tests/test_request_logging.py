"""Request IDs (X-Request-ID), structured logging and the per-request lifecycle log lines.

App log records are captured with the application's own filter and JSON formatter attached
to the "app" logger, so each captured line is exactly what would be printed.
"""

import asyncio
import http.client
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid

import anyio
import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from conftest import BACKEND_DIR, delete_committed_application
from test_chat_streaming import (
    BODY as STREAM_BODY,
    ESTIMATE,
    HANG,
    SECRET_DETAIL,
    URL,
    Gate,
    ScriptedStream,
    asgi_scope,
    assert_settled_once,
    events,
    finished,
    run_asgi,
)

from app.api.middleware import RequestContextMiddleware
from app.core import request_context
from app.core.config import Settings
from app.core.structured_logging import (
    JsonFormatter,
    RequestContextFilter,
    TextFormatter,
    configure_logging,
    log_event,
)
from app.database.connection import SessionLocal, get_db
from app.database.models import UsageRecord, UsageStatus
from app.main import app
from app.providers.base import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    InvalidProviderRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.providers.fake import FAKE_MODEL, FakeProvider
from app.services import quota as quota_service
from app.services.api_keys import create_api_key, create_application

BODY = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
SUMMARY = "inference_request_finished"
HTTP_FINISHED = "http_request_finished"


class LogCapture(logging.Handler):
    """Formats records exactly like the application's handler, and keeps the lines."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.addFilter(RequestContextFilter())
        self.setFormatter(JsonFormatter())
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))

    @property
    def records(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]

    def event(self, name) -> list[dict]:
        return [r for r in self.records if r.get("event") == name]

    def summary(self) -> dict:
        """The only inference_request_finished record (fails unless there is exactly one)."""
        [record] = self.event(SUMMARY)
        return record

    def http_finished(self) -> dict:
        [record] = self.event(HTTP_FINISHED)
        return record

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture
def app_logs():
    logger = logging.getLogger("app")
    handler = LogCapture()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


@pytest.fixture
def client(client, auth_headers):
    client.headers.update(auth_headers)
    return client


@pytest.fixture
def lenient_client(db_session, auth_headers):
    """A client that returns 500 responses instead of re-raising the server's exception."""
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app, raise_server_exceptions=False, headers=auth_headers) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)


def is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except (TypeError, ValueError):
        return False


def start_headers(sent) -> dict[str, str]:
    start = next(m for m in sent.messages if m["type"] == "http.response.start")
    return {k.decode().lower(): v.decode() for k, v in start["headers"]}


class FailingChat:
    """A provider whose chat() raises a non-provider exception (a bug)."""

    name = "buggy"

    def __init__(self, exc):
        self.exc = exc

    async def chat(self, request):
        raise self.exc

    async def aclose(self):
        pass


# --- Request IDs ---


def test_a_missing_request_id_is_generated_and_returned(client, app_logs):
    response = client.get("/v1/models")

    request_id = response.headers["X-Request-ID"]
    assert response.status_code == 200 and is_uuid(request_id)
    record = app_logs.http_finished()
    # Without a client ID the external ID is the internal UUID's string form.
    assert record["request_id"] == request_id == record["gateway_request_id"]
    assert (record["method"], record["route"], record["status_code"]) == ("GET", "/v1/models", 200)
    assert isinstance(record["duration_ms"], float)


@pytest.mark.parametrize("value", ["a", "A" * 64, "req-1.2_3:4", "0", "-._:", "550e8400-e29b-41d4-a716-446655440000"])
def test_a_valid_request_id_is_preserved(client, app_logs, value):
    response = client.get("/v1/models", headers={"X-Request-ID": value})

    assert response.headers["X-Request-ID"] == value
    record = app_logs.http_finished()
    assert record["request_id"] == value
    # The internal ID is always generated, even when the client's value is itself a UUID.
    assert is_uuid(record["gateway_request_id"]) and record["gateway_request_id"] != value


@pytest.mark.parametrize("value", ["", "A" * 65, "has space", "slash/id", "semi;colon", "quote\"", "comma,id", "tab\tid", "plus+id"])
def test_an_invalid_request_id_is_replaced(client, app_logs, value):
    response = client.get("/v1/models", headers={"X-Request-ID": value})

    request_id = response.headers["X-Request-ID"]
    assert request_id != value and is_uuid(request_id)
    assert app_logs.http_finished()["request_id"] == request_id


def test_request_id_validation_boundaries():
    assert request_context.valid_client_request_id("x" * 64) == "x" * 64
    assert request_context.valid_client_request_id("x" * 65) is None
    assert request_context.valid_client_request_id("x") == "x"
    assert request_context.valid_client_request_id("") is None
    assert request_context.valid_client_request_id(None) is None
    assert request_context.valid_client_request_id("abc\n") is None  # fullmatch: no trailing newline
    assert request_context.valid_client_request_id("ä") is None


def test_only_the_first_request_id_header_counts(client):
    response = client.get("/v1/models", headers=[("X-Request-ID", "first"), ("X-Request-ID", "second")])

    assert response.headers.get_list("X-Request-ID") == ["first"]


def test_every_response_carries_the_request_id(client, app_logs):
    for path, status in [("/health", 200), ("/v1/models", 200), ("/no-such-route", 404)]:
        request_id = f"id-{status}-{path.strip('/').replace('/', '.')}"
        response = client.get(path, headers={"X-Request-ID": request_id})
        assert response.status_code == status
        assert response.headers["X-Request-ID"] == request_id
    unmatched = [r for r in app_logs.event(HTTP_FINISHED) if r["status_code"] == 404]
    assert [r["route"] for r in unmatched] == [None]  # never the raw path


def test_the_internal_id_is_the_database_key_and_the_completion_id(client, db_session, api_key, app_logs):
    response = client.post(URL, json=BODY, headers={"X-Request-ID": "client-chosen"})

    assert response.headers["X-Request-ID"] == "client-chosen"
    summary = app_logs.summary()
    gateway_id = uuid.UUID(summary["gateway_request_id"])
    assert response.json()["id"] == f"chatcmpl-{gateway_id.hex}"
    db_session.expire_all()
    [record] = db_session.scalars(select(UsageRecord).where(UsageRecord.application_id == api_key.record.application_id))
    assert record.request_id == gateway_id


def test_a_reused_client_request_id_never_collides_in_the_database(client, db_session, api_key):
    """Clients reuse their IDs (retries): the unique usage and reservation keys don't see them."""
    responses = [client.post(URL, json=BODY, headers={"X-Request-ID": "retry-me"}) for _ in range(3)]

    assert [r.status_code for r in responses] == [200] * 3
    assert all(r.headers["X-Request-ID"] == "retry-me" for r in responses)
    assert len({r.json()["id"] for r in responses}) == 3
    db_session.expire_all()
    rows = db_session.scalars(select(UsageRecord).where(UsageRecord.application_id == api_key.record.application_id))
    assert len({row.request_id for row in rows}) == 3


def test_the_middleware_isolates_concurrent_requests_and_resets_the_context():
    """Each request's IDs are visible in its own task, in tasks it starts and in worker
    threads, never in another request's, and are gone once it returns."""
    seen = {}

    async def inner_app(scope, receive, send):
        marker = scope["path"]
        own = request_context.current()
        await anyio.sleep(0.01 * (hash(marker) % 5))  # interleave the requests
        in_child = []
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: _record(in_child))
        in_thread = await anyio.to_thread.run_sync(request_context.current)
        seen[marker] = (own, request_context.current(), in_child[0], in_thread)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-request-id", b"set-by-app")]})
        await send({"type": "http.response.body", "body": b""})

    async def _record(target):
        await anyio.sleep(0)
        target.append(request_context.current())

    middleware = RequestContextMiddleware(inner_app)
    headers_sent = {}

    async def one(i):
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {"type": "http", "method": "GET", "path": f"/r{i}", "headers": [(b"x-request-id", f"req-{i}".encode())]}
        await middleware(scope, receive, send)
        headers_sent[i] = sent[0]["headers"]

    async def scenario():
        async with anyio.create_task_group() as tg:
            for i in range(20):
                tg.start_soon(one, i)
        return request_context.current()

    after = anyio.run(scenario)

    assert after is None
    for i in range(20):
        own, later, in_child, in_thread = seen[f"/r{i}"]
        assert own.request_id == f"req-{i}" and own == later == in_child == in_thread
        # The app's own X-Request-ID is replaced, not duplicated.
        assert headers_sent[i] == [(b"x-request-id", f"req-{i}".encode())]
    assert len({seen[f"/r{i}"][0].gateway_request_id for i in range(20)}) == 20


class MarkedProvider:
    """Logs the marker in each request's prompt from inside the provider call (and, for
    streams, from inside the streaming task), so the log's request IDs can be checked."""

    name = "marked"
    logger = logging.getLogger("app.tests.marked_provider")

    async def chat(self, request):
        marker = request.messages[0].content
        await asyncio.sleep(0.005 * (int(marker) % 4))
        self.logger.info("provider handling %s", marker)
        return ChatResponse(request.model, ChatMessage("assistant", "ok"), "stop", Usage(1, 1))

    async def stream(self, request):
        marker = request.messages[0].content
        yield ChatChunk("")
        await asyncio.sleep(0.005 * (int(marker) % 4))
        self.logger.info("provider handling %s", marker)
        yield ChatChunk("ok")
        yield ChatChunk("", "stop", Usage(1, 1))

    async def aclose(self):
        pass


@pytest.fixture
def committed_key():
    name = f"reqlog-{uuid.uuid4().hex[:8]}"
    with SessionLocal() as db:
        application = create_application(db, name, token_limit=100_000)
        _, plaintext = create_api_key(db, application, "reqlog")
        application_id = application.id
        db.commit()
    yield plaintext, application_id
    with SessionLocal() as db:
        delete_committed_application(db, name)


def test_concurrent_requests_do_not_leak_ids_into_each_others_logs(committed_key, use_provider, app_logs):
    plaintext, application_id = committed_key
    use_provider(MarkedProvider())
    count = 8

    async def one(i):
        body = json.dumps({**BODY, "messages": [{"role": "user", "content": str(i)}], "stream": i % 2 == 0}).encode()
        state = {"sent": False}
        sent = []

        async def receive():
            if not state["sent"]:
                state["sent"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(3600)

        async def send(message):
            sent.append(message)

        headers = {"authorization": f"Bearer {plaintext}", "x-request-id": f"client-{i}"}
        await app(asgi_scope(body, headers, "2.3"), receive, send)
        return sent

    with TestClient(app) as client:
        async def scenario():
            results = {}

            async def run(i):
                results[i] = await one(i)

            with anyio.fail_after(20):
                async with anyio.create_task_group() as tg:
                    for i in range(count):
                        tg.start_soon(run, i)
            return results

        results = client.portal.call(scenario)

    gateway_ids = {}
    for i, sent in results.items():
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert start["status"] == 200 and (b"x-request-id", f"client-{i}".encode()) in start["headers"]
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()
        completion_id = events(body)[0]["id"] if i % 2 == 0 else json.loads(body)["id"]
        [summary] = [r for r in app_logs.event(SUMMARY) if r["request_id"] == f"client-{i}"]
        assert summary["stream"] is (i % 2 == 0) and summary["outcome"] == "completed"
        gateway_ids[i] = uuid.UUID(summary["gateway_request_id"])
        assert completion_id == f"chatcmpl-{gateway_ids[i].hex}"
        [provider_line] = [r for r in app_logs.records if r["message"] == f"provider handling {i}"]
        assert (provider_line["request_id"], provider_line["gateway_request_id"]) == (f"client-{i}", str(gateway_ids[i]))
        for record in app_logs.records:
            if record.get("request_id") == f"client-{i}":
                assert record["gateway_request_id"] == str(gateway_ids[i])
    assert len(app_logs.event(SUMMARY)) == count
    with SessionLocal() as db:
        stored = set(db.scalars(select(UsageRecord.request_id).where(UsageRecord.application_id == application_id)))
    assert stored == set(gateway_ids.values())


# --- Errors ---


def test_handled_http_errors_carry_the_request_id(client, auth_headers, app_logs):
    unknown = client.post(URL, json={**BODY, "model": "no-such-model"}, headers={"X-Request-ID": "unknown-model"})
    unauthorized = client.post(URL, json=BODY, headers={"Authorization": "Bearer if_wrong", "X-Request-ID": "no-auth"})
    invalid = client.post(URL, json={"model": FAKE_MODEL}, headers={"X-Request-ID": "invalid-body"})

    assert (unknown.status_code, unknown.headers["X-Request-ID"]) == (404, "unknown-model")
    assert (unauthorized.status_code, unauthorized.headers["X-Request-ID"]) == (401, "no-auth")
    assert (invalid.status_code, invalid.headers["X-Request-ID"]) == (422, "invalid-body")
    assert unknown.json()["detail"]["code"] == "model_not_found"  # response bodies unchanged
    # The previously invisible INFO rejections are logged, with the request's ID.
    assert any(r["request_id"] == "unknown-model" and "Rejected request" in r["message"] for r in app_logs.records)
    assert any(r["request_id"] == "no-auth" and "invalid or revoked API key" in r["message"] for r in app_logs.records)
    # Requests rejected by the gateway have no inference summary and no outcome at all, so
    # they can never be counted as provider_rejected.
    assert app_logs.event(SUMMARY) == []
    assert not any("outcome" in r for r in app_logs.records)
    assert {r["request_id"]: r["status_code"] for r in app_logs.event(HTTP_FINISHED)} == {
        "unknown-model": 404, "no-auth": 401, "invalid-body": 422,
    }


def test_insufficient_quota_carries_the_request_id(client, app_logs):
    body = {**BODY, "messages": [{"role": "user", "content": "x" * 2_000_000}]}

    response = client.post(URL, json=body, headers={"X-Request-ID": "no-quota"})

    assert (response.status_code, response.headers["X-Request-ID"]) == (429, "no-quota")
    assert response.json()["detail"]["code"] == "insufficient_quota"
    assert app_logs.event(SUMMARY) == []  # rejected before the provider stage


def test_an_unhandled_error_is_a_500_with_the_request_id(lenient_client, use_provider, app_logs, api_key):
    use_provider(FailingChat(RuntimeError(f"bug near {SECRET_DETAIL}")))

    response = lenient_client.post(URL, json=BODY, headers={"X-Request-ID": "server-bug"})

    assert response.status_code == 500 and response.headers["X-Request-ID"] == "server-bug"
    assert response.text == "Internal Server Error"  # Starlette's response, unchanged
    assert SECRET_DETAIL not in response.text
    summary = app_logs.summary()
    assert (summary["outcome"], summary["http_status"], summary["settlement"]) == ("gateway_error", 500, "not_attempted")
    finished_record = app_logs.http_finished()
    assert finished_record["level"] == "ERROR" and finished_record["status_code"] == 500
    assert finished_record["request_id"] == "server-bug" and finished_record["error_type"] == "RuntimeError"
    assert "RuntimeError" in finished_record["exception"]  # the traceback, with the request's IDs
    assert api_key.plaintext not in app_logs.text


def test_the_server_still_sees_the_unhandled_exception(use_provider, db_session, auth_headers):
    """The middleware doesn't swallow exceptions: the server (here the test client) gets it."""
    use_provider(FailingChat(RuntimeError("bug")))
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app, headers=auth_headers) as test_client:
            with pytest.raises(RuntimeError, match="bug"):
                test_client.post(URL, json=BODY)
    finally:
        app.dependency_overrides.pop(get_db, None)


# --- Non-streaming lifecycle summaries ---


def test_a_completed_request_logs_one_summary(client, app_logs):
    response = client.post(URL, json=BODY, headers={"X-Request-ID": "ok-1"})

    assert response.status_code == 200
    summary = app_logs.summary()
    assert summary["request_id"] == "ok-1" and summary["level"] == "INFO"
    assert {k: summary[k] for k in ("provider", "model", "stream", "outcome", "http_status", "usage_reported", "settlement")} == {
        "provider": "fake", "model": FAKE_MODEL, "stream": False, "outcome": "completed", "http_status": 200,
        "usage_reported": True, "settlement": "succeeded",
    }
    assert (summary["prompt_tokens"], summary["completion_tokens"]) == (1, 4)
    assert "client_disconnected" not in summary and summary["duration_ms"] >= 0


@pytest.mark.parametrize(
    "error, status, outcome",
    [
        (ProviderTimeoutError("slow"), 504, "timeout"),
        (ProviderUnavailableError("down"), 503, "provider_error"),
        (InvalidProviderRequestError("too long"), 400, "provider_rejected"),
        (ModelNotFoundError("missing"), 502, "provider_error"),
        (ProviderError("odd"), 502, "provider_error"),
    ],
)
def test_provider_failures_log_their_outcome(client, use_provider, app_logs, error, status, outcome):
    use_provider(FakeProvider(fail_with=error))

    response = client.post(URL, json=BODY)

    assert response.status_code == status
    summary = app_logs.summary()
    assert (summary["outcome"], summary["http_status"], summary["settlement"], summary["usage_reported"]) == (
        outcome, status, "succeeded", False,
    )
    assert summary["request_id"] == response.headers["X-Request-ID"]


def test_a_settlement_failure_is_in_the_summary(client, monkeypatch, app_logs):
    def failing_settle(*args, **kwargs):
        raise OperationalError("UPDATE", {}, Exception("connection lost"))

    monkeypatch.setattr(quota_service, "settle", failing_settle)

    assert client.post(URL, json=BODY).status_code == 200  # behaviour unchanged
    summary = app_logs.summary()
    assert (summary["outcome"], summary["settlement"]) == ("completed", "failed")


def test_a_cancelled_non_streaming_request_logs_cancelled(client, use_provider, auth_headers, app_logs):
    use_provider(FakeProvider(delay_seconds=3600))
    raw = json.dumps(BODY).encode()
    state = {"sent": False}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await asyncio.sleep(3600)

    async def scenario():
        async def send(message):
            pass

        with anyio.move_on_after(0.3):
            await app(asgi_scope(raw, auth_headers, "2.3"), receive, send)

    client.portal.call(scenario)

    summary = app_logs.summary()
    assert (summary["outcome"], summary["http_status"], summary["settlement"]) == ("cancelled", None, "not_attempted")
    finished_record = app_logs.http_finished()
    assert finished_record["error_type"] == "CancelledError" and finished_record["level"] == "INFO"


# --- Streaming ---


def test_a_completed_stream_keeps_its_ids_and_logs_one_summary(client, use_provider, db_session, api_key, app_logs):
    use_provider(ScriptedStream([ChatChunk("a"), finished(Usage(1, 1))]))

    response = client.post(URL, json=STREAM_BODY, headers={"X-Request-ID": "stream-ok"})

    assert response.headers["X-Request-ID"] == "stream-ok"
    assert events(response.text)[-1] == "[DONE]"
    summary = app_logs.summary()
    assert {k: summary[k] for k in ("outcome", "stream", "stream_completed", "client_disconnected", "settlement", "http_status")} == {
        "outcome": "completed", "stream": True, "stream_completed": True, "client_disconnected": False,
        "settlement": "succeeded", "http_status": 200,
    }
    assert events(response.text)[0]["id"] == f"chatcmpl-{uuid.UUID(summary['gateway_request_id']).hex}"
    assert all(r.get("request_id") == "stream-ok" for r in app_logs.records)
    assert app_logs.http_finished()["status_code"] == 200
    assert_settled_once(db_session, api_key.record.application_id, charged=2, status=UsageStatus.SUCCEEDED, reported=True)


@pytest.mark.parametrize(
    "error, outcome",
    [(ProviderTimeoutError(SECRET_DETAIL), "timeout"), (ProviderUnavailableError(SECRET_DETAIL), "provider_error")],
)
def test_a_failure_after_acceptance_keeps_the_correlation_id(client, use_provider, db_session, api_key, app_logs, error, outcome):
    use_provider(ScriptedStream([ChatChunk("a"), error]))

    response = client.post(URL, json=STREAM_BODY, headers={"X-Request-ID": "mid-stream"})

    assert response.status_code == 200 and response.headers["X-Request-ID"] == "mid-stream"
    assert "error" in events(response.text)[-1]
    [warning] = app_logs.event("provider_request_failed")
    assert warning["request_id"] == "mid-stream" and warning["phase"] == "mid_stream"
    summary = app_logs.summary()
    assert summary["request_id"] == "mid-stream" and summary["gateway_request_id"] == warning["gateway_request_id"]
    # HTTP said 200; the summary says what actually happened.
    assert (summary["outcome"], summary["http_status"], summary["stream_completed"], summary["level"]) == (
        outcome, 200, False, "WARNING",
    )
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_a_gateway_bug_mid_stream_is_a_gateway_error(client, use_provider, app_logs):
    use_provider(ScriptedStream([ChatChunk("a"), RuntimeError("bug")]))

    client.post(URL, json=STREAM_BODY)

    summary = app_logs.summary()
    assert (summary["outcome"], summary["http_status"], summary["level"]) == ("gateway_error", 200, "ERROR")


@pytest.mark.parametrize(
    "error, status, outcome",
    [
        (ProviderTimeoutError("slow"), 504, "timeout"),
        (ProviderUnavailableError("down"), 503, "provider_error"),
        (InvalidProviderRequestError("too long"), 400, "provider_rejected"),
    ],
)
def test_failures_before_acceptance_log_their_status(client, use_provider, app_logs, error, status, outcome):
    use_provider(ScriptedStream(before_accept=error))

    response = client.post(URL, json=STREAM_BODY)

    assert response.status_code == status
    summary = app_logs.summary()
    assert (summary["outcome"], summary["http_status"], summary["stream_completed"], summary["client_disconnected"]) == (
        outcome, status, False, False,
    )


def test_a_stream_that_yields_nothing_is_a_provider_error(client, use_provider, app_logs):
    use_provider(ScriptedStream(accept=False))

    assert client.post(URL, json=STREAM_BODY).status_code == 502
    assert (app_logs.summary()["outcome"], app_logs.summary()["http_status"]) == ("provider_error", 502)


def test_a_provider_bug_before_acceptance_is_a_500_with_the_request_id(lenient_client, use_provider, app_logs):
    use_provider(ScriptedStream(before_accept=RuntimeError("bug in provider")))

    response = lenient_client.post(URL, json=STREAM_BODY, headers={"X-Request-ID": "accept-bug"})

    assert (response.status_code, response.headers["X-Request-ID"]) == (500, "accept-bug")
    summary = app_logs.summary()
    assert (summary["outcome"], summary["http_status"], summary["request_id"]) == ("gateway_error", 500, "accept-bug")


def test_a_client_disconnect_logs_one_summary(client, use_provider, db_session, api_key, auth_headers, app_logs):
    disconnect = Gate()
    provider = use_provider(ScriptedStream([ChatChunk("a"), HANG]))

    def on_send(message):
        if b'"content":"a"' in message.get("body", b""):
            disconnect.set()

    sent = run_asgi(client, STREAM_BODY, {**auth_headers, "X-Request-ID": "gone"}, disconnect=disconnect, on_send=on_send)

    assert start_headers(sent)["x-request-id"] == "gone"
    summary = app_logs.summary()
    assert (summary["outcome"], summary["client_disconnected"], summary["stream_completed"], summary["http_status"]) == (
        "client_disconnected", True, False, 200,
    )
    assert summary["request_id"] == "gone" and summary["settlement"] == "succeeded"
    assert app_logs.http_finished()["request_id"] == "gone"
    assert provider.closed == 1
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_a_disconnect_while_blocked_in_send_logs_one_summary(client, use_provider, auth_headers, app_logs):
    disconnect = Gate()
    use_provider(ScriptedStream([ChatChunk("a"), ChatChunk("b"), finished()]))

    async def slow_send(message):
        if b'"content":"a"' in message.get("body", b""):
            disconnect.set()
            await asyncio.sleep(3600)

    run_asgi(client, STREAM_BODY, auth_headers, disconnect=disconnect, send_hook=slow_send)

    assert app_logs.summary()["outcome"] == "client_disconnected"


def test_a_disconnect_while_waiting_for_acceptance_is_a_client_disconnect(client, use_provider, auth_headers, app_logs):
    use_provider(ScriptedStream([ChatChunk("a"), HANG], accept_delay=0.2))

    run_asgi(client, STREAM_BODY, auth_headers, disconnect_immediately=True)

    summary = app_logs.summary()
    assert (summary["outcome"], summary["client_disconnected"]) == ("client_disconnected", True)


def test_a_failing_send_under_spec_2_4_is_a_client_disconnect(client, use_provider, auth_headers, app_logs):
    use_provider(ScriptedStream([ChatChunk("a"), ChatChunk("b"), finished()]))

    async def failing_send(message):
        if b'"content":"a"' in message.get("body", b""):
            raise OSError("connection reset")

    with pytest.raises(Exception):  # ClientDisconnect, as before
        run_asgi(client, STREAM_BODY, auth_headers, send_hook=failing_send, spec_version="2.4")

    summary = app_logs.summary()
    assert (summary["outcome"], summary["client_disconnected"], summary["http_status"]) == ("client_disconnected", True, 200)
    finished_record = app_logs.http_finished()
    assert finished_record["error_type"] == "ClientDisconnect" and finished_record["level"] == "INFO"


def test_cancellation_is_not_a_client_disconnect(client, use_provider, db_session, api_key, auth_headers, app_logs):
    """E.g. a server shutdown cancelling the request task: no http.disconnect was received."""
    use_provider(ScriptedStream([ChatChunk("a"), HANG]))
    raw = json.dumps(STREAM_BODY).encode()
    state = {"sent": False}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await asyncio.sleep(3600)

    async def scenario():
        async def send(message):
            pass

        with anyio.move_on_after(0.5):
            await app(asgi_scope(raw, {**auth_headers, "X-Request-ID": "shutdown"}, "2.3"), receive, send)

    client.portal.call(scenario)

    summary = app_logs.summary()
    assert (summary["outcome"], summary["client_disconnected"], summary["request_id"]) == ("cancelled", False, "shutdown")
    assert_settled_once(db_session, api_key.record.application_id, charged=ESTIMATE, status=UsageStatus.FAILED, reported=False)


def test_a_client_disconnect_through_a_real_uvicorn_server_is_classified(committed_key, use_provider, app_logs):
    """uvicorn's h11 server reports the closed socket as http.disconnect (ASGI spec 2.3)."""
    plaintext, _ = committed_key
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
        body = json.dumps(STREAM_BODY).encode()
        with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
            conn.sendall(
                f"POST {URL} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {plaintext}\r\n"
                f"X-Request-ID: real-socket\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            )
            received = b""
            while b'"content":"first"' not in received:
                part = conn.recv(65536)
                assert part, "server closed the connection early"
                received += part
        assert b"x-request-id: real-socket" in received.lower()
        deadline = time.monotonic() + 5
        while not app_logs.event(SUMMARY) and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        server.should_exit = True
        thread.join(10)

    summary = app_logs.summary()
    assert (summary["outcome"], summary["client_disconnected"], summary["request_id"]) == (
        "client_disconnected", True, "real-socket",
    )
    assert summary["settlement"] == "succeeded" and provider.closed == 1


def test_a_settlement_failure_in_a_stream_is_in_the_summary(client, use_provider, monkeypatch, app_logs):
    def failing_settle(*args, **kwargs):
        raise OperationalError("UPDATE", {}, Exception("connection lost"))

    monkeypatch.setattr(quota_service, "settle", failing_settle)
    use_provider(ScriptedStream([ChatChunk("a"), finished()]))

    assert events(client.post(URL, json=STREAM_BODY).text)[-1] == "[DONE]"
    summary = app_logs.summary()
    assert (summary["outcome"], summary["settlement"]) == ("completed", "failed")


def test_an_unexpected_settlement_bug_in_a_stream_is_in_the_summary(client, use_provider, monkeypatch, app_logs):
    monkeypatch.setattr(quota_service, "settle", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug")))
    use_provider(ScriptedStream([ChatChunk("a"), finished()]))

    client.post(URL, json=STREAM_BODY)

    assert app_logs.summary()["settlement"] == "failed"


# --- What is (not) logged ---


@pytest.mark.parametrize("stream, phase", [(False, "response"), (True, "before_accept")])
@pytest.mark.parametrize(
    "error, upstream_status, outcome",
    [
        (InvalidProviderRequestError(SECRET_DETAIL, status_code=400), 400, "provider_rejected"),
        (InvalidProviderRequestError(SECRET_DETAIL, status_code=413), 413, "provider_rejected"),
        (ProviderTimeoutError(SECRET_DETAIL, status_code=504), 504, "timeout"),
        (ProviderUnavailableError(SECRET_DETAIL), None, "provider_error"),
        (ProviderError(SECRET_DETAIL, status_code=401), 401, "provider_error"),  # gateway's own key refused
    ],
)
def test_provider_failures_are_logged_as_metadata_only(client, use_provider, app_logs, stream, phase, error, upstream_status, outcome):
    provider = ScriptedStream(before_accept=error) if stream else FakeProvider(fail_with=error)
    use_provider(provider)

    response = client.post(URL, json={**BODY, "stream": stream}, headers={"X-Request-ID": "upstream-fail"})

    [failure] = app_logs.event("provider_request_failed")
    assert {k: failure[k] for k in ("level", "request_id", "provider", "model", "stream", "phase", "error_type", "upstream_status", "outcome")} == {
        "level": "WARNING", "request_id": "upstream-fail", "provider": provider.name, "model": FAKE_MODEL,
        "stream": stream, "phase": phase, "error_type": type(error).__name__, "upstream_status": upstream_status,
        "outcome": outcome,
    }
    assert failure["gateway_request_id"] == app_logs.summary()["gateway_request_id"]
    assert app_logs.summary()["outcome"] == outcome and app_logs.summary()["http_status"] == response.status_code
    for leaked in (SECRET_DETAIL, "10.1.2.3", "sk-live", "upstream http"):
        assert leaked not in app_logs.text


def test_a_registry_mismatch_is_an_error_with_the_configured_model_id(client, use_provider, app_logs):
    use_provider(FakeProvider(fail_with=ModelNotFoundError(SECRET_DETAIL, status_code=404)), models={FAKE_MODEL: "upstream-name"})

    assert client.post(URL, json=BODY).status_code == 502
    [failure] = app_logs.event("provider_request_failed")
    assert failure["level"] == "ERROR" and failure["provider_model_id"] == "upstream-name"
    assert "check the model registry" in failure["message"] and failure["outcome"] == "provider_error"
    assert SECRET_DETAIL not in app_logs.text


def test_outcome_names_are_the_documented_set():
    from typing import get_args

    from app.api.inference_log import Outcome

    assert set(get_args(Outcome)) == {
        "completed", "provider_error", "timeout", "provider_rejected", "gateway_error", "client_disconnected", "cancelled",
    }


def test_no_secrets_prompts_or_completions_are_logged(client, api_key, app_logs):
    prompt = "PROMPT-CONTENT-7731"
    body = {**BODY, "messages": [{"role": "user", "content": prompt}]}

    completion = client.post(URL, json=body).json()["choices"][0]["message"]["content"]
    streamed = client.post(URL, json={**body, "stream": True}).text
    client.post(URL, json=body, headers={"Authorization": "Bearer if_not_a_real_key_123456"})

    assert prompt in completion and prompt in streamed  # the fake provider echoes it
    assert len(app_logs.event(SUMMARY)) == 2
    for forbidden in (prompt, api_key.plaintext, "if_not_a_real_key_123456", "Bearer", "Fake response"):
        assert forbidden not in app_logs.text


# --- Logging configuration and format ---


def test_json_lines_have_the_documented_shape():
    handler = LogCapture()
    logger = logging.getLogger("app.tests.format")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        ids = request_context.new_request_ids("abc")
        token = request_context.bind(ids)
        try:
            log_event(logger, logging.INFO, "something_happened", "Something", count=3, message="ignored", when=object())
            try:
                raise ValueError("boom")
            except ValueError:
                logger.exception("Failed")
        finally:
            request_context.unbind(token)
        logger.warning("Outside a request")
    finally:
        logger.removeHandler(handler)

    event, failure, outside = handler.records
    assert event["timestamp"].endswith("Z") and event["level"] == "INFO" and event["logger"] == "app.tests.format"
    assert (event["event"], event["message"], event["count"]) == ("something_happened", "Something", 3)  # fields never overwrite
    assert (event["request_id"], event["gateway_request_id"]) == ("abc", str(ids.gateway_request_id))
    assert isinstance(event["when"], str)  # anything not JSON-serializable is stringified
    assert "ValueError: boom" in failure["exception"] and "Traceback" in failure["exception"]
    assert "request_id" not in outside and "event" not in outside


def test_text_format():
    record = logging.LogRecord("app.x", logging.INFO, __file__, 1, "Hello %s", ("world",), None)
    record.event, record.fields = "greeted", {"who": "world"}
    token = request_context.bind(request_context.new_request_ids("t-1"))
    try:
        RequestContextFilter().filter(record)
    finally:
        request_context.unbind(token)

    line = TextFormatter().format(record)

    assert " INFO app.x [request_id=t-1 gateway_request_id=" in line
    assert line.endswith('Hello world event=greeted who="world"')


def test_configure_logging_is_idempotent_and_leaves_uvicorn_alone():
    app_logger = logging.getLogger("app")
    uvicorn_handlers = list(logging.getLogger("uvicorn").handlers)
    try:
        configure_logging("WARNING", "text")
        configure_logging("DEBUG", "json")
        ours = [h for h in app_logger.handlers if type(h).__name__ == "_StdoutHandler"]
        assert len(ours) == 1 and isinstance(ours[0].formatter, JsonFormatter)
        assert app_logger.level == logging.DEBUG and app_logger.propagate
        assert logging.getLogger("uvicorn").handlers == uvicorn_handlers
    finally:
        from app.core.config import settings

        configure_logging(settings.log_level, settings.log_format)


def test_log_settings_are_case_insensitive_and_validated():
    configured = Settings(_env_file=None, database_url="postgresql://unused", log_level="debug", log_format="TEXT")
    assert (configured.log_level, configured.log_format) == ("DEBUG", "text")
    with pytest.raises(ValueError):
        Settings(_env_file=None, database_url="postgresql://unused", log_format="xml")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_info_logs_are_visible_under_uvicorns_logging_configuration():
    """Starts the real uvicorn CLI (its default logging configuration) on a free loopback port,
    with the built-in fake registry and the test database and Redis (from this process's
    environment), and checks the app's INFO lines reach stdout as JSON."""
    port = _free_port()
    env = {
        **os.environ,
        "MODEL_REGISTRY_FILE": "",  # the built-in fake registry: no provider is ever contacted
        "CHAT_PROVIDER": "fake",
        "SCHEDULER_ENABLED": "false",
        "INFERENCE_ENGINE": "simulated",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json",
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=BACKEND_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                connection.request("GET", "/health", headers={"X-Request-ID": "uvicorn-check"})
                response = connection.getresponse()
                break
            except ConnectionRefusedError:
                assert process.poll() is None, process.stderr.read()
                assert time.monotonic() < deadline, "uvicorn did not start"
                time.sleep(0.1)
        assert response.status == 200 and response.getheader("X-Request-ID") == "uvicorn-check"
        connection.close()
    finally:
        process.terminate()
        stdout, stderr = process.communicate(timeout=20)

    lines = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    assert any(r["level"] == "INFO" and r["logger"] == "app.providers.registry" for r in lines), stdout
    [finished_record] = [r for r in lines if r.get("event") == HTTP_FINISHED]
    assert (finished_record["request_id"], finished_record["route"], finished_record["status_code"]) == ("uvicorn-check", "/health", 200)
    assert "Application startup complete" in stderr  # uvicorn's own logging is unchanged
