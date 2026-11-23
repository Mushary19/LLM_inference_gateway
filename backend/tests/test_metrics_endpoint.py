"""GET /metrics: disabled by default, bearer-token protected, isolated registries, and a
scrape never touches providers, PostgreSQL, Redis, API keys, rate limits or quotas."""

import json
import logging
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, Counter
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr, ValidationError
from sqlalchemy import event

from conftest import BACKEND_DIR

from app.api import metrics as metrics_api
from app.core.config import METRICS_TOKEN_MIN_LENGTH, Settings, settings
from app.core.metrics import CONTENT_TYPE, GatewayMetrics, metrics_of
from app.core.structured_logging import JsonFormatter, RequestContextFilter
from app.database.connection import engine
from app.main import InferenceFlowApp, app
from app.services.rate_limit import TokenBucketRateLimiter

TOKEN = "metrics-token-" + "x" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "metrics_token", SecretStr(TOKEN))


@pytest.fixture
def probe_counter(isolated_metrics):
    """A metric in this test's registry, to see what a scrape exposes."""
    counter = Counter("inferenceflow_test_probe", "Test-only probe.", registry=isolated_metrics.registry)
    counter.inc(3)
    return counter


@pytest.fixture
def scrape_client():
    with TestClient(app) as client:
        yield client


def settings_with(**values) -> Settings:
    return Settings(_env_file=None, database_url="postgresql://unused", **values)


# --- Disabled (the default) ---


def test_disabled_by_default_is_a_plain_404(scrape_client, probe_counter):
    assert settings.metrics_token is None
    response = scrape_client.get("/metrics", headers=AUTH)

    assert response.status_code == 404
    assert response.json() == scrape_client.get("/no-such-path").json() == {"detail": "Not Found"}
    assert "inferenceflow_test_probe" not in response.text


def test_an_empty_token_disables_the_endpoint(monkeypatch, scrape_client, probe_counter):
    for value in ("", "   "):
        monkeypatch.setattr(settings, "metrics_token", settings_with(metrics_token=value).metrics_token)
        assert settings.metrics_token is None
        response = scrape_client.get("/metrics", headers={"Authorization": "Bearer "})
        assert response.status_code == 404 and "inferenceflow" not in response.text


# --- Enabled ---


def test_a_valid_token_scrapes_the_apps_registry(enabled, scrape_client, probe_counter):
    response = scrape_client.get("/metrics", headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE
    assert response.headers["X-Request-ID"]  # the request ID middleware still applies
    families = {f.name: f for f in text_string_to_metric_families(response.text)}
    [sample] = families["inferenceflow_test_probe"].samples
    assert sample.value == 3


def test_the_scheme_is_case_insensitive(enabled, scrape_client):
    assert scrape_client.get("/metrics", headers={"Authorization": f"bearer {TOKEN}"}).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": ""},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": f"Bearer {TOKEN}x"},
        {"Authorization": f"Bearer {TOKEN[:-1]}"},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": TOKEN},
        {"Authorization": f"Bearer  {TOKEN}"},
        {"Authorization": f"Token {TOKEN}"},
        {"Authorization": "Bearer if_an_api_key_is_not_a_metrics_token"},
    ],
    ids=["missing", "empty", "scheme-only", "scheme-space", "longer", "shorter", "wrong", "basic",
         "no-scheme", "double-space", "other-scheme", "api-key"],
)
def test_unauthorized_requests_get_no_metrics(enabled, scrape_client, probe_counter, headers):
    response = scrape_client.get("/metrics", headers=headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json() == {"detail": "Missing or invalid metrics token"}
    assert "inferenceflow_test_probe" not in response.text and TOKEN not in response.text


def test_the_token_is_compared_in_constant_time(enabled, scrape_client, monkeypatch):
    calls = []
    real = metrics_api.hmac.compare_digest

    def recording(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(metrics_api.hmac, "compare_digest", recording)

    assert scrape_client.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert scrape_client.get("/metrics", headers=AUTH).status_code == 200
    assert calls == [(b"nope", TOKEN.encode()), (TOKEN.encode(), TOKEN.encode())]


def test_not_in_the_openapi_schema_or_the_model_list(enabled, client, auth_headers):
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]
    models = client.get("/v1/models", headers=auth_headers).json()["data"]
    assert all("metrics" not in model["id"] for model in models)


# --- Configuration ---


def test_a_strong_token_is_accepted_and_masked():
    configured = settings_with(metrics_token=f"  {TOKEN}  ")

    assert configured.metrics_token.get_secret_value() == TOKEN  # surrounding whitespace dropped
    assert TOKEN not in repr(configured) and TOKEN not in str(configured.metrics_token)


def test_unset_or_empty_means_disabled():
    assert settings_with().metrics_token is None
    assert settings_with(metrics_token="").metrics_token is None
    assert settings_with(metrics_token=" \t ").metrics_token is None


@pytest.mark.parametrize("length", [1, 16, METRICS_TOKEN_MIN_LENGTH - 1])
def test_a_short_token_fails_without_echoing_it(length):
    weak = "w" * length
    with pytest.raises(ValidationError) as caught:
        settings_with(metrics_token=weak)

    message = str(caught.value)
    assert f"METRICS_TOKEN must be at least {METRICS_TOKEN_MIN_LENGTH} characters" in message
    assert weak not in message


def test_exactly_the_minimum_length_is_accepted():
    assert settings_with(metrics_token="m" * METRICS_TOKEN_MIN_LENGTH).metrics_token is not None


def test_a_short_token_stops_application_startup():
    """Settings are loaded when the application is imported, so a weak METRICS_TOKEN makes
    the process fail at startup rather than serve with it."""
    weak = "short-metrics-token"
    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=BACKEND_DIR, env={**os.environ, "METRICS_TOKEN": weak}, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "METRICS_TOKEN must be at least" in result.stderr and weak not in result.stderr + result.stdout


def test_the_token_never_appears_in_logs(enabled, scrape_client, caplog):
    logger = logging.getLogger("app")
    lines = []
    handler = logging.Handler()
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(JsonFormatter())
    handler.emit = lambda record: lines.append(handler.format(record))
    logger.addHandler(handler)
    try:
        with caplog.at_level(logging.DEBUG):
            scrape_client.get("/metrics", headers=AUTH)
            scrape_client.get("/metrics", headers={"Authorization": f"Bearer {TOKEN}-wrong"})
    finally:
        logger.removeHandler(handler)

    assert any(json.loads(line).get("route") == "/metrics" for line in lines)  # the scrapes were logged
    assert TOKEN not in "\n".join(lines) and TOKEN not in caplog.text


# --- Registry isolation ---


def test_each_application_instance_has_its_own_registry():
    first, second = InferenceFlowApp(), InferenceFlowApp()

    assert metrics_of(first) is not metrics_of(second)
    assert metrics_of(first).registry is not metrics_of(second).registry
    # The same metric name in both: no duplicate-registration error, independent values.
    a = Counter("inferenceflow_same_name", "x", registry=metrics_of(first).registry)
    b = Counter("inferenceflow_same_name", "x", registry=metrics_of(second).registry)
    a.inc(2)
    b.inc(5)
    assert metrics_of(first).registry.get_sample_value("inferenceflow_same_name_total") == 2
    assert metrics_of(second).registry.get_sample_value("inferenceflow_same_name_total") == 5


def test_nothing_uses_the_global_default_registry():
    before = {family.name for family in REGISTRY.collect()}
    GatewayMetrics()
    InferenceFlowApp()
    assert {family.name for family in REGISTRY.collect()} == before


def test_each_test_starts_with_an_empty_registry(isolated_metrics, enabled, scrape_client):
    """The autouse fixture installs a fresh GatewayMetrics on the app for every test; the
    probe counter other tests register is not here."""
    assert metrics_of(app) is isolated_metrics
    response = scrape_client.get("/metrics", headers=AUTH)
    assert "inferenceflow_test_probe" not in response.text
    assert list(text_string_to_metric_families(response.text)) == []


def test_the_endpoint_serves_the_registry_currently_on_the_app(enabled, scrape_client):
    replacement = GatewayMetrics()
    Counter("inferenceflow_replacement_probe", "x", registry=replacement.registry).inc()
    app.state.metrics = replacement  # the autouse isolated_metrics fixture restores the original

    assert "inferenceflow_replacement_probe_total 1.0" in scrape_client.get("/metrics", headers=AUTH).text


# --- A scrape is local and free ---


class Untouchable:
    """A provider that fails the test if anything calls it."""

    name = "untouchable"

    def __getattr__(self, item):
        raise AssertionError(f"provider.{item} was used during a scrape")


def test_scraping_touches_no_provider_database_or_redis(enabled, use_provider, monkeypatch, probe_counter):
    use_provider(Untouchable())
    checkouts = []
    listener = lambda *args: checkouts.append(args)  # noqa: E731
    event.listen(engine, "checkout", listener)

    async def no_redis(self, *args, **kwargs):
        raise AssertionError("Redis was used during a scrape")

    monkeypatch.setattr(TokenBucketRateLimiter, "consume", no_redis)
    try:
        with TestClient(app) as client:
            responses = [client.get("/metrics", headers=AUTH) for _ in range(3)]
            unauthorized = client.get("/metrics")
    finally:
        event.remove(engine, "checkout", listener)

    assert [r.status_code for r in responses] == [200] * 3 and unauthorized.status_code == 401
    assert checkouts == []  # no PostgreSQL connection (so no API key lookup and no quota)
    assert "X-RateLimit-Limit" not in responses[0].headers  # the rate limiter never ran


def test_scraping_does_not_change_the_metrics(enabled, scrape_client, probe_counter):
    """A scrape records nothing itself, so it can't count as traffic, now or once HTTP
    metrics exist (the /metrics route is to be excluded from them)."""
    first = scrape_client.get("/metrics", headers=AUTH).text
    for _ in range(3):
        scrape_client.get("/metrics", headers=AUTH)
        scrape_client.get("/metrics")
    assert scrape_client.get("/metrics", headers=AUTH).text == first
