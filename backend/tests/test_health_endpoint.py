"""GET /health: liveness only. No authentication, no rate limiting, no dependencies."""

import pytest
from fastapi.testclient import TestClient

from app.api.rate_limit import get_rate_limiter
from app.database.connection import get_db
from app.main import app


class ExplodingProvider:
    name = "exploding"

    def __getattr__(self, attribute):
        raise AssertionError(f"GET /health must not use the provider ({attribute})")


@pytest.fixture
def isolated_client(use_provider):
    """The app with every dependency /health must not touch replaced by something that fails."""

    def must_not_be_used():  # no parameters, or FastAPI would treat them as query fields
        raise AssertionError("GET /health must not use this dependency")

    use_provider(ExplodingProvider())
    app.dependency_overrides[get_db] = must_not_be_used
    app.dependency_overrides[get_rate_limiter] = must_not_be_used
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_rate_limiter, None)


def test_liveness_needs_no_key_and_touches_nothing(isolated_client):
    response = isolated_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-RateLimit-Limit" not in response.headers


def test_liveness_ignores_credentials(isolated_client):
    response = isolated_client.get("/health", headers={"Authorization": "Bearer garbage"})

    assert response.status_code == 200


def test_liveness_survives_an_unreachable_redis(monkeypatch):
    """Redis down makes /v1 fail closed (503) but leaves liveness alone."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(settings, "redis_timeout_seconds", 0.2)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_liveness_is_public_in_the_openapi_spec(isolated_client):
    operation = isolated_client.get("/openapi.json").json()["paths"]["/health"]["get"]

    assert "security" not in operation


def test_v1_routes_are_still_protected(client):
    assert client.get("/v1/models").status_code == 401
