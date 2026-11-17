"""Rate limiting of the /v1 API: ordering with authentication, responses, failures, lifecycle.

A spy wraps the app's real limiter (real Lua script on test-redis), so tests can prove when
the limiter is, and is not, reached.
"""

import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager

import pytest
import redis
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import event

from conftest import delete_committed_application

from app.api.chat import get_chat_provider, router as v1_router
from app.api.rate_limit import get_rate_limiter
from app.core.config import settings
from app.database.connection import SessionLocal, engine, get_db
from app.main import app, create_rate_limiter
from app.providers.base import ChatMessage, ChatResponse, ModelInfo, ProviderUnavailableError, Usage
from app.providers.fake import FAKE_MODEL, FakeProvider
from app.services.api_keys import create_api_key, create_application, generate_api_key, revoke_api_key
from app.services.rate_limit import REDIS_CLIENT_NAME, TokenBucketRateLimiter, bucket_key, create_redis_client

CHAT = ("POST", "/v1/chat/completions")
MODELS = ("GET", "/v1/models")
PROTECTED = [CHAT, MODELS]
CHAT_BODY = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "hi"}]}


def send(client, endpoint, headers=None, json=CHAT_BODY):
    method, url = endpoint
    if method == "POST":
        return client.post(url, json=json, headers=headers or {})
    return client.get(url, headers=headers or {})


class CountingProvider:
    name = "counting"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def chat(self, request):
        self.calls.append("chat")
        return ChatResponse(FAKE_MODEL, ChatMessage("assistant", "ok"), "stop", Usage(1, 1))

    async def stream(self, request):
        self.calls.append("stream")
        yield  # pragma: no cover

    async def list_models(self):
        self.calls.append("list_models")
        return [ModelInfo(id=FAKE_MODEL, owned_by="fake")]


class SpyLimiter:
    """Delegates to a real TokenBucketRateLimiter and records every call."""

    def __init__(self, make_inner) -> None:
        self._make_inner = make_inner
        self._inner = None
        self.calls: list[tuple[str, int | None]] = []

    def bind(self, request: Request) -> "SpyLimiter":
        if self._inner is None:  # built on the app's event loop, reusing its Redis client
            self._inner = self._make_inner(request.app.state.rate_limiter)
        return self

    async def consume(self, bucket_id, *, now_us=None):
        self.calls.append((bucket_id, now_us))
        return await self._inner.consume(bucket_id, now_us=now_us)


@pytest.fixture
def provider():
    counting = CountingProvider()
    app.dependency_overrides[get_chat_provider] = lambda: counting
    yield counting
    app.dependency_overrides.pop(get_chat_provider, None)


@pytest.fixture
def limiter():
    """Install a spy limiter; by default it wraps the app's own limiter (60/min, burst 10)."""
    installed = {}

    def install(make_inner=lambda real: real) -> SpyLimiter:
        spy = SpyLimiter(make_inner)
        installed["spy"] = spy
        app.dependency_overrides[get_rate_limiter] = spy.bind
        return spy

    install()
    yield install
    app.dependency_overrides.pop(get_rate_limiter, None)


@pytest.fixture
def spy(limiter):
    return limiter()


def tight(burst: int, rpm: int = 60):
    """A real limiter with a small bucket, sharing the app's Redis client."""
    return lambda real: TokenBucketRateLimiter(real.redis, rpm, burst)


def redis_admin() -> redis.Redis:
    """A plain client to inspect test-redis (TEST_REDIS_URL, via the guard)."""
    return redis.Redis.from_url(settings.redis_url)


# --- Authentication runs first; rejected keys never reach Redis ---


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
@pytest.mark.parametrize(
    "authorization",
    [None, "Basic dXNlcjpwYXNz", "Bearer", "Bearer garbage", "Bearer if_short"],
    ids=["missing", "basic", "bearer-only", "garbage", "malformed"],
)
def test_missing_or_malformed_keys_never_reach_the_limiter(client, provider, spy, endpoint, authorization):
    headers = {} if authorization is None else {"Authorization": authorization}

    response = send(client, endpoint, headers)

    assert response.status_code == 401 and response.json() == {"detail": "Missing or invalid API key"}
    assert "X-RateLimit-Limit" not in response.headers
    assert spy.calls == [] and provider.calls == []


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
def test_unknown_and_revoked_keys_never_reach_the_limiter(client, provider, spy, db_session, api_key, endpoint):
    revoke_api_key(db_session, api_key.record.key_prefix)
    db_session.commit()

    for plaintext in (generate_api_key().plaintext, api_key.plaintext):
        response = send(client, endpoint, {"Authorization": f"Bearer {plaintext}"})
        assert response.status_code == 401

    assert spy.calls == [] and provider.calls == []
    assert redis_admin().exists(bucket_key(str(api_key.record.id))) == 0


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
def test_valid_key_is_checked_against_its_own_bucket(client, provider, spy, api_key, auth_headers, endpoint):
    response = send(client, endpoint, auth_headers)

    assert response.status_code == 200
    # One check, for this key's ID, with Redis time (the route never injects a timestamp).
    assert spy.calls == [(str(api_key.record.id), None)]
    assert redis_admin().exists(bucket_key(str(api_key.record.id))) == 1
    assert api_key.plaintext not in str(redis_admin().keys("inferenceflow:ratelimit:*"))


def test_authentication_and_rate_limit_run_before_body_validation(client, provider, limiter, auth_headers):
    spy = limiter(tight(burst=1))
    assert send(client, CHAT, auth_headers).status_code == 200

    limited = send(client, CHAT, auth_headers, json={"not": "valid"})

    assert limited.status_code == 429  # not 422: the bucket is checked before the body
    assert len(spy.calls) == 2 and provider.calls == ["chat"]


def test_key_is_looked_up_once_per_request(client, provider, spy, auth_headers):
    lookups = []
    listener = lambda conn, cur, statement, *a: lookups.append(statement) if "FROM api_keys" in statement else None  # noqa: E731
    event.listen(engine, "before_cursor_execute", listener)
    try:
        assert send(client, CHAT, auth_headers).status_code == 200
    finally:
        event.remove(engine, "before_cursor_execute", listener)

    assert len(lookups) == 1  # require_api_key is shared by the router, the limiter and the route


# --- Accepted and rejected responses ---


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
def test_accepted_requests_carry_rate_limit_headers(client, provider, spy, auth_headers, endpoint):
    first = send(client, endpoint, auth_headers)
    second = send(client, endpoint, auth_headers)

    assert first.headers["X-RateLimit-Limit"] == "10"
    assert first.headers["X-RateLimit-Remaining"] == "9"
    assert first.headers["X-RateLimit-Reset"] == "1"  # 1 token short, refilled at 1/s
    assert second.headers["X-RateLimit-Remaining"] == "8"
    assert "Retry-After" not in first.headers


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
def test_exceeding_the_limit_returns_429_without_calling_the_provider(client, provider, limiter, auth_headers, endpoint):
    spy = limiter(tight(burst=2, rpm=30))  # 0.5 tokens/s

    responses = [send(client, endpoint, auth_headers) for _ in range(4)]

    assert [r.status_code for r in responses] == [200, 200, 429, 429]
    rejected = responses[2]
    assert rejected.json() == {"detail": "Rate limit exceeded"}
    assert rejected.headers["Retry-After"] == "2"  # one token at 0.5/s, rounded up
    assert rejected.headers["X-RateLimit-Limit"] == "2"
    assert rejected.headers["X-RateLimit-Remaining"] == "0"
    assert rejected.headers["X-RateLimit-Reset"] == "4"
    assert len(provider.calls) == 2 and len(spy.calls) == 4


def test_keys_have_independent_buckets(client, provider, limiter, db_session, auth_headers):
    limiter(tight(burst=1))
    other_app = create_application(db_session, "other-app")
    _, other_key = create_api_key(db_session, other_app, "other")
    db_session.commit()
    other_headers = {"Authorization": f"Bearer {other_key}"}

    assert send(client, CHAT, auth_headers).status_code == 200
    assert send(client, CHAT, auth_headers).status_code == 429
    assert send(client, CHAT, other_headers).status_code == 200


@pytest.fixture
def committed_key():
    """A key committed to the test database, for tests that need the app's real per-request
    sessions (the shared rollback session can't serve concurrent requests). Removed afterwards."""
    name = f"ratelimit-concurrency-{uuid.uuid4().hex[:8]}"
    with SessionLocal() as db:
        _, plaintext = create_api_key(db, create_application(db, name), "concurrency")
        db.commit()
    try:
        yield plaintext
    finally:
        with SessionLocal() as db:
            delete_committed_application(db, name)


def test_concurrent_http_requests_admit_exactly_the_burst(provider, limiter, committed_key):
    limiter(tight(burst=5, rpm=1))
    headers = {"Authorization": f"Bearer {committed_key}"}
    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(20)

    with TestClient(app) as client:

        def call():
            barrier.wait()
            status = send(client, MODELS, headers).status_code
            with lock:
                results.append(status)

        threads = [threading.Thread(target=call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert sorted(results) == [200] * 5 + [429] * 15
    assert provider.calls == ["list_models"] * 5


def test_provider_errors_keep_their_mapping(client, spy, auth_headers):
    app.dependency_overrides[get_chat_provider] = lambda: FakeProvider(fail_with=ProviderUnavailableError("upstream down"))
    try:
        response = send(client, CHAT, auth_headers)
    finally:
        app.dependency_overrides.pop(get_chat_provider, None)

    assert response.status_code == 503 and response.json() == {"detail": "upstream down"}
    assert len(spy.calls) == 1  # the request was admitted, then the provider failed


# --- Redis failures: fail closed with 503 ---


@pytest.fixture
def closed_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def stalled_port():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    held, stop = [], threading.Event()

    def accept():
        server.settimeout(0.1)
        while not stop.is_set():
            try:
                held.append(server.accept()[0])
            except OSError:
                pass

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    thread.join()
    for connection in held:
        connection.close()
    server.close()


def limiter_at(port: int, timeout: float):
    return lambda real: TokenBucketRateLimiter(create_redis_client(f"redis://127.0.0.1:{port}/0", timeout), 60, 10)


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
@pytest.mark.parametrize("failure", ["unreachable", "timeout"])
def test_redis_failure_returns_503_and_never_calls_the_provider(
    client, provider, limiter, auth_headers, closed_port, stalled_port, endpoint, failure
):
    port = closed_port if failure == "unreachable" else stalled_port
    spy = limiter(limiter_at(port, timeout=0.2))

    response = send(client, endpoint, auth_headers)

    assert response.status_code == 503
    assert response.json() == {"detail": "Rate limiting is temporarily unavailable"}
    body = response.text.lower()
    assert "redis" not in body and "127.0.0.1" not in body and str(port) not in body
    assert len(spy.calls) == 1 and provider.calls == []


def test_redis_failure_is_checked_before_body_validation(client, provider, limiter, auth_headers, closed_port):
    limiter(limiter_at(closed_port, timeout=0.2))

    response = send(client, CHAT, auth_headers, json={"not": "valid"})

    assert response.status_code == 503 and provider.calls == []


# --- Coverage of the router ---


def test_new_v1_routes_inherit_authentication_and_rate_limiting(db_session, auth_headers, limiter):
    spy = limiter(tight(burst=1))

    @v1_router.get("/inheritance-probe")
    async def probe():
        return {"reached": True}

    probe_route = v1_router.routes[-1]

    @asynccontextmanager
    async def probe_lifespan(probe_app):
        # Its own limiter, on its own event loop, exactly as the real app builds one.
        probe_app.state.rate_limiter = create_rate_limiter()
        try:
            yield
        finally:
            await probe_app.state.rate_limiter.aclose()

    try:
        probe_app = FastAPI(lifespan=probe_lifespan)
        probe_app.include_router(v1_router)
        probe_app.dependency_overrides = {**app.dependency_overrides, get_db: lambda: db_session}
        with TestClient(probe_app) as probe_client:
            unauthenticated = probe_client.get("/v1/inheritance-probe")
            first = probe_client.get("/v1/inheritance-probe", headers=auth_headers)
            second = probe_client.get("/v1/inheritance-probe", headers=auth_headers)
    finally:
        v1_router.routes.remove(probe_route)

    assert unauthenticated.status_code == 401
    assert first.status_code == 200 and first.json() == {"reached": True}
    assert first.headers["X-RateLimit-Limit"] == "1"
    assert second.status_code == 429
    assert len(spy.calls) == 2


def test_generate_is_unchanged_and_not_rate_limited(client, spy):
    response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 5})

    assert response.status_code == 202
    assert "X-RateLimit-Limit" not in response.headers
    assert spy.calls == []


# --- Redis client lifecycle ---


def app_connection_ids() -> set[str]:
    """IDs of test-redis connections opened by the app's rate-limit client (by client name)."""
    return {c["id"] for c in redis_admin().client_list() if c.get("name") == REDIS_CLIENT_NAME}


def wait_until_closed(ids: set[str], timeout: float = 2.0) -> set[str]:
    """Redis processes a client's disconnect asynchronously; wait (bounded) for these IDs to go."""
    deadline = time.monotonic() + timeout
    while (remaining := ids & app_connection_ids()) and time.monotonic() < deadline:
        time.sleep(0.01)
    return remaining


def test_one_redis_client_is_reused_and_closed_on_shutdown(db_session, auth_headers):
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        before = app_connection_ids()
        with TestClient(app) as client:
            limiter = app.state.rate_limiter
            pool = limiter.redis.connection_pool
            assert pool.max_connections == 50
            assert pool.connection_kwargs["socket_timeout"] == settings.redis_timeout_seconds
            assert pool.connection_kwargs["socket_connect_timeout"] == settings.redis_timeout_seconds
            assert app_connection_ids() - before == set()  # creating the client doesn't connect
            for _ in range(5):
                assert send(client, MODELS, auth_headers).status_code == 200
            opened = app_connection_ids() - before
            assert app.state.rate_limiter is limiter
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert len(opened) == 1  # five sequential requests shared one pooled connection
    assert wait_until_closed(opened) == set()  # shutdown closed it


def test_shutdown_survives_a_failing_redis_close(monkeypatch, caplog):
    with TestClient(app):
        limiter = app.state.rate_limiter
        provider = app.state.chat_provider

        async def broken_aclose():
            raise ConnectionError("close failed")

        monkeypatch.setattr(limiter, "aclose", broken_aclose)
        monkeypatch.setattr(provider, "aclose", provider.aclose)

    assert "Failed to close the Redis rate-limit client" in caplog.text
