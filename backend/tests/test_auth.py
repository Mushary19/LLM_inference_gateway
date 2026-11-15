"""Authentication of the /v1 endpoints (Authorization: Bearer <api_key>)."""

import asyncio
import logging
import uuid

import psycopg
import pytest
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.api.auth import AuthenticatedClient, require_api_key
from app.api.chat import get_chat_provider
from app.database.connection import SessionLocal, engine
from app.database.models import ApiKey, ClientApplication
from app.main import app
from app.providers.base import ChatMessage, ChatResponse, ModelInfo, ProviderUnavailableError, Usage
from app.providers.fake import FAKE_MODEL
from app.services.api_keys import create_api_key, create_application, generate_api_key, revoke_api_key

CHAT = ("POST", "/v1/chat/completions")
MODELS = ("GET", "/v1/models")
PROTECTED = [CHAT, MODELS]
CHAT_BODY = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "hi"}]}
EXPECTED_401 = {"detail": "Missing or invalid API key"}


class CountingProvider:
    """Records every provider call, so tests can prove rejected requests never reach it."""

    name = "counting"

    def __init__(self, list_models_error=None) -> None:
        self.calls: list[str] = []
        self.list_models_error = list_models_error

    async def chat(self, request):
        self.calls.append("chat")
        return ChatResponse(FAKE_MODEL, ChatMessage("assistant", "ok"), "stop", Usage(1, 1))

    async def stream(self, request):
        self.calls.append("stream")
        yield  # pragma: no cover

    async def list_models(self):
        self.calls.append("list_models")
        if self.list_models_error is not None:
            raise self.list_models_error
        return [ModelInfo(id=FAKE_MODEL, owned_by="fake")]

    async def health(self):
        self.calls.append("health")

    async def aclose(self):
        pass


@pytest.fixture
def provider():
    counting = CountingProvider()
    app.dependency_overrides[get_chat_provider] = lambda: counting
    yield counting
    app.dependency_overrides.pop(get_chat_provider, None)


def send(client, endpoint, headers=None, json=CHAT_BODY):
    method, url = endpoint
    if method == "POST":
        return client.post(url, json=json, headers=headers or {})
    return client.get(url, headers=headers or {})


def assert_rejected(response):
    assert response.status_code == 401
    assert response.json() == EXPECTED_401
    assert response.headers["www-authenticate"] == "Bearer"


# --- Rejections ---


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
@pytest.mark.parametrize(
    "authorization",
    [
        None,  # missing header
        "",
        "Basic dXNlcjpwYXNz",  # wrong scheme
        "Token if_abcdefgh_" + "a" * 43,  # wrong scheme, well-formed key
        "Bearer",  # no token
        "Bearer ",
        "Bearer garbage",  # malformed
        "Bearer if_short",
        "Bearer if_abcdefgh_" + "a" * 43 + " extra",
    ],
    ids=["missing", "empty", "basic", "token-scheme", "bearer-only", "bearer-space", "garbage", "short", "extra"],
)
def test_missing_or_malformed_credentials_are_rejected(client, provider, endpoint, authorization):
    headers = {} if authorization is None else {"Authorization": authorization}

    assert_rejected(send(client, endpoint, headers))
    assert provider.calls == []


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
def test_unknown_well_formed_key_is_rejected(client, provider, endpoint):
    unknown = generate_api_key().plaintext

    assert_rejected(send(client, endpoint, {"Authorization": f"Bearer {unknown}"}))
    assert provider.calls == []


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
def test_revoked_key_is_rejected_like_an_unknown_key(client, provider, db_session, api_key, endpoint):
    revoke_api_key(db_session, api_key.record.key_prefix)
    db_session.commit()

    revoked = send(client, endpoint, {"Authorization": f"Bearer {api_key.plaintext}"})
    unknown = send(client, endpoint, {"Authorization": f"Bearer {generate_api_key().plaintext}"})

    assert_rejected(revoked)
    assert (revoked.json(), dict(revoked.headers)["www-authenticate"]) == (
        unknown.json(),
        dict(unknown.headers)["www-authenticate"],
    )
    assert provider.calls == []


def test_revocation_blocks_subsequent_requests(client, provider, db_session, api_key, auth_headers):
    assert send(client, CHAT, auth_headers).status_code == 200

    revoke_api_key(db_session, api_key.record.key_prefix)
    db_session.commit()

    assert_rejected(send(client, CHAT, auth_headers))
    assert_rejected(send(client, MODELS, auth_headers))
    assert provider.calls == ["chat"]


def test_authentication_runs_before_body_validation(client, provider):
    response = send(client, CHAT, {}, json={"not": "a chat request"})

    assert_rejected(response)
    assert provider.calls == []


# --- Accepted requests ---


def test_valid_key_reaches_chat_completions(client, provider, auth_headers):
    response = send(client, CHAT, auth_headers)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert provider.calls == ["chat"]


def test_valid_key_reaches_list_models(client, provider, auth_headers):
    response = send(client, MODELS, auth_headers)

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": FAKE_MODEL, "object": "model", "created": 0, "owned_by": "fake"}],
    }
    assert provider.calls == ["list_models"]


def test_bearer_scheme_is_case_insensitive(client, provider, api_key):
    response = send(client, CHAT, {"Authorization": f"bearer {api_key.plaintext}"})

    assert response.status_code == 200


def test_list_models_maps_provider_errors(client, auth_headers):
    app.dependency_overrides[get_chat_provider] = lambda: CountingProvider(
        list_models_error=ProviderUnavailableError("upstream down")
    )
    try:
        response = send(client, MODELS, auth_headers)
    finally:
        app.dependency_overrides.pop(get_chat_provider, None)

    assert response.status_code == 503 and response.json() == {"detail": "upstream down"}


# --- Authenticated client reaches the route; keys never logged ---


def test_dependency_resolves_the_client_application(db_session, api_key):
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=api_key.plaintext)

    client_info = require_api_key(credentials, db_session)

    assert client_info == AuthenticatedClient(
        api_key_id=api_key.record.id,
        key_prefix=api_key.record.key_prefix,
        key_name="test-key",
        application_id=api_key.record.application_id,
        application_name="test-app",
    )


def test_route_receives_the_authenticated_client(client, provider, api_key, auth_headers, caplog):
    with caplog.at_level(logging.INFO, logger="app.api.chat"):
        assert send(client, CHAT, auth_headers).status_code == 200

    route_log = [r.getMessage() for r in caplog.records if r.name == "app.api.chat"]
    assert route_log == [f"Chat completion for application test-app (key {api_key.record.key_prefix}), model {FAKE_MODEL}"]


def test_full_keys_are_never_logged(client, provider, api_key, auth_headers, caplog):
    unknown = generate_api_key().plaintext
    with caplog.at_level(logging.DEBUG):
        send(client, CHAT, auth_headers)
        send(client, CHAT, {"Authorization": f"Bearer {unknown}"})
        send(client, MODELS, {"Authorization": "Bearer garbage-secret-value"})

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert api_key.plaintext not in logged
    assert unknown not in logged and "garbage-secret-value" not in logged


# --- Scope ---


def test_legacy_generate_endpoint_is_not_protected(client):
    response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 5})

    assert response.status_code == 202


def test_openapi_declares_bearer_auth_on_v1_routes_only(client):
    spec = client.get("/openapi.json").json()

    assert spec["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    assert spec["paths"]["/v1/chat/completions"]["post"]["security"] == [{"HTTPBearer": []}]
    assert spec["paths"]["/v1/models"]["get"]["security"] == [{"HTTPBearer": []}]
    assert "security" not in spec["paths"]["/generate"]["post"]


# --- Regression: authentication must not hold a database connection during the provider call ---
# These use the app's real per-request get_db (connection pool), not the shared rollback
# session, so they commit a key to the test database and remove it afterwards.


@pytest.fixture
def committed_key():
    application_name = f"pool-regression-{uuid.uuid4().hex[:8]}"
    with SessionLocal() as db:
        application = create_application(db, application_name)
        _, plaintext = create_api_key(db, application, "pool-regression")
        db.commit()
    try:
        yield plaintext
    finally:
        with SessionLocal() as db:
            application_id = db.scalar(
                select(ClientApplication.id).where(ClientApplication.name == application_name)
            )
            db.execute(delete(ApiKey).where(ApiKey.application_id == application_id))
            db.execute(delete(ClientApplication).where(ClientApplication.id == application_id))
            db.commit()


def _idle_in_transaction_backends() -> int:
    """Count this database's sessions sitting idle inside an open transaction (outside the pool)."""
    url = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(url) as connection:
        return connection.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
            "AND state LIKE 'idle in transaction%' AND pid <> pg_backend_pid()"
        ).fetchone()[0]


class PoolObservingProvider:
    """Records the connection pool and server-side transaction state while a call is in progress."""

    name = "pool-observing"

    def __init__(self) -> None:
        self.observations: list[tuple[int, int]] = []

    async def _observe(self) -> None:
        checked_out = engine.pool.checkedout()
        idle_in_transaction = await asyncio.to_thread(_idle_in_transaction_backends)
        self.observations.append((checked_out, idle_in_transaction))

    async def chat(self, request):
        await self._observe()
        return ChatResponse(FAKE_MODEL, ChatMessage("assistant", "ok"), "stop", Usage(1, 1))

    async def list_models(self):
        await self._observe()
        return [ModelInfo(id=FAKE_MODEL, owned_by="fake")]


@pytest.mark.parametrize("endpoint", PROTECTED, ids=["chat", "models"])
def test_connection_is_returned_to_the_pool_before_the_provider_call(committed_key, endpoint):
    provider = PoolObservingProvider()
    app.dependency_overrides[get_chat_provider] = lambda: provider
    try:
        with TestClient(app) as client:
            assert engine.pool.checkedout() == 0
            response = send(client, endpoint, {"Authorization": f"Bearer {committed_key}"})
    finally:
        app.dependency_overrides.pop(get_chat_provider, None)

    assert response.status_code == 200
    # While the provider was running: no pooled connection checked out, and no session left
    # idle in the authentication's read transaction.
    assert provider.observations == [(0, 0)]


def test_authenticated_client_does_not_depend_on_the_session(committed_key):
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=committed_key)
    with SessionLocal() as db:
        client_info = require_api_key(credentials, db)

        assert not db.in_transaction()  # the read transaction was ended
    # The session is closed now; every field is a plain value, not a lazy ORM attribute.
    assert isinstance(client_info.api_key_id, uuid.UUID)
    assert isinstance(client_info.application_id, uuid.UUID)
    assert client_info.key_prefix == committed_key[:11]
    assert client_info.key_name == "pool-regression"
    assert client_info.application_name.startswith("pool-regression-")
