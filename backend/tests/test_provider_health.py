"""Provider health and model availability checks (app/services/provider_health.py).

Real OpenAICompatibleProvider instances talk to mocked upstreams (httpx MockTransport), so
the HTTP-to-error mapping under test is the production one. No network is used.
"""

import asyncio
import logging
import time

import httpx2 as httpx
import pytest

from app.providers.base import (
    ChatMessage,
    ChatRequest,
    HealthStatus,
    ModelInfo,
    ProviderError,
    ProviderUnavailableError,
)
from app.providers.fake import FAKE_MODEL, FakeProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.registry import ModelRegistry, ModelRoute
from app.services.provider_health import (
    HealthReport,
    ModelCheck,
    ProviderCheck,
    check_registry,
    classify_failure,
    probe_provider,
)

BASE_URL = "http://internal-llm.corp.example:8000/v1"
SECRET = "sk-health-secret-0123456789"


def models_body(*ids):
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "x"} for model_id in ids]}


class Upstream:
    """Mock upstream: records requests; answers with a response, raises, or hangs."""

    def __init__(self, response=None, raises=None, hang_seconds=None):
        self.response = response if response is not None else httpx.Response(200, json=models_body("m1"))
        self.raises = raises
        self.hang_seconds = hang_seconds
        self.requests: list[httpx.Request] = []
        self.cancelled = False

    async def __call__(self, request):
        self.requests.append(request)
        if self.hang_seconds is not None:
            try:
                await asyncio.sleep(self.hang_seconds)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.raises is not None:
            raise self.raises
        return self.response


def provider_for(upstream, name="upstream", api_key=SECRET, timeout_seconds=120.0):
    return OpenAICompatibleProvider(
        BASE_URL, api_key=api_key, timeout_seconds=timeout_seconds,
        transport=httpx.MockTransport(upstream), name=name,
    )


def probe(provider, timeout=2.0) -> ProviderCheck:
    async def scenario():
        try:
            return await probe_provider(provider.name, provider, timeout)
        finally:
            await provider.aclose()

    return asyncio.run(scenario())


def check(registry, timeout=2.0) -> HealthReport:
    async def scenario():
        try:
            return await check_registry(registry, timeout)
        finally:
            await registry.aclose()

    return asyncio.run(scenario())


# --- Provider health: one probe, classified into fixed reason codes ---


def test_healthy_provider_reports_its_listing():
    upstream = Upstream(httpx.Response(200, json=models_body("a", "b")))

    result = probe(provider_for(upstream))

    assert (result.status, result.reason) == ("healthy", None)
    assert result.listed_models == frozenset({"a", "b"})
    assert result.latency_ms >= 0


@pytest.mark.parametrize(
    "upstream, status, reason",
    [
        (Upstream(raises=httpx.ConnectError("[Errno 111] Connection refused")), "unhealthy", "connection_failed"),
        (Upstream(raises=httpx.ReadError("connection reset by peer")), "unhealthy", "connection_failed"),
        (Upstream(raises=httpx.ConnectTimeout("connect timed out")), "unhealthy", "timeout"),
        (Upstream(raises=httpx.ReadTimeout("read timed out")), "unhealthy", "timeout"),
        (Upstream(httpx.Response(401, json={"error": {"message": "Invalid API Key"}})), "unhealthy", "authentication_failed"),
        (Upstream(httpx.Response(403, json={"error": {"message": "forbidden"}})), "unhealthy", "authentication_failed"),
        (Upstream(httpx.Response(404, text="404 page not found")), "unverified", "listing_not_supported"),
        (Upstream(httpx.Response(400, json={"error": {"message": "bad"}})), "unhealthy", "request_rejected"),
        (Upstream(httpx.Response(422, json={"error": {"message": "bad"}})), "unhealthy", "unexpected_status"),
        (Upstream(httpx.Response(405, text="method not allowed")), "unhealthy", "unexpected_status"),
        (Upstream(httpx.Response(418, text="teapot")), "unhealthy", "unexpected_status"),
        (Upstream(httpx.Response(408, text="request timeout")), "unhealthy", "timeout"),
        (Upstream(httpx.Response(429, json={"error": {"message": "slow down"}})), "unhealthy", "rate_limited"),
        (Upstream(httpx.Response(500, text="Internal Server Error")), "unhealthy", "server_error"),
        (Upstream(httpx.Response(502, text="Bad Gateway")), "unhealthy", "server_error"),
        (Upstream(httpx.Response(503, text="Service Unavailable")), "unhealthy", "server_error"),
        (Upstream(httpx.Response(504, text="Gateway Timeout")), "unhealthy", "timeout"),
        (Upstream(httpx.Response(200, text="<html>captive portal</html>")), "unhealthy", "invalid_response"),
        (Upstream(httpx.Response(200, json={"models": ["m1"]})), "unhealthy", "invalid_response"),
        (Upstream(httpx.Response(200, json={"data": [{"name": "no id"}]})), "unhealthy", "invalid_response"),
    ],
    ids=[
        "refused", "reset", "connect-timeout", "read-timeout", "401", "403", "404", "400", "422", "405",
        "418", "408", "429", "500", "502", "503", "504", "html", "wrong-shape", "missing-id",
    ],
)
def test_failures_are_classified(upstream, status, reason):
    result = probe(provider_for(upstream))

    assert (result.status, result.reason) == (status, reason)
    assert result.listed_models == frozenset()


def test_unexpected_exceptions_are_contained():
    class BrokenProvider(FakeProvider):
        async def list_models(self):
            raise RuntimeError(f"bug while talking to {BASE_URL} with {SECRET}")

    result = probe(BrokenProvider(name="broken"))

    assert (result.status, result.reason) == ("unhealthy", "probe_error")
    assert SECRET not in repr(result) and "internal-llm" not in repr(result)


def test_a_malformed_listing_from_a_custom_provider_is_contained():
    class NoneListing(FakeProvider):
        async def list_models(self):
            return None  # not iterable

    assert probe(NoneListing(name="none")).reason == "probe_error"


@pytest.mark.parametrize(
    "exc, expected",
    [
        (asyncio.TimeoutError(), ("unhealthy", "timeout")),
        (ProviderError("unreadable"), ("unhealthy", "invalid_response")),
        (ProviderUnavailableError("refused"), ("unhealthy", "connection_failed")),
        (ProviderError("x", status_code=599), ("unhealthy", "server_error")),
        (ProviderError("x", status_code=302), ("unhealthy", "unexpected_status")),
        (ValueError("x"), ("unhealthy", "probe_error")),
    ],
)
def test_classification_fallbacks(exc, expected):
    assert classify_failure(exc) == expected


# --- Timeouts and cancellation ---


def test_a_hung_provider_times_out_and_its_request_is_cancelled():
    upstream = Upstream(hang_seconds=30)
    provider = provider_for(upstream, timeout_seconds=120)  # the inference timeout is irrelevant

    started = time.monotonic()
    result = probe(provider, timeout=0.2)
    elapsed = time.monotonic() - started

    assert (result.status, result.reason) == ("unhealthy", "timeout")
    assert elapsed < 2
    assert upstream.cancelled  # the in-flight request was cancelled, not left running
    assert provider._client.timeout == httpx.Timeout(120)  # the provider's own timeout is untouched


def test_probe_timeout_does_not_shorten_inference_requests():
    upstream = Upstream(hang_seconds=0.3)
    upstream.response = httpx.Response(
        200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "model": "m1"}
    )
    provider = provider_for(upstream)

    async def scenario():
        try:
            probe_result = await probe_provider("p", provider, 0.05)
            # A chat request right after the probe still gets the provider's full timeout.
            response = await provider.chat(ChatRequest("m1", (ChatMessage("user", "hi"),)))
            return probe_result, response
        finally:
            await provider.aclose()

    probe_result, response = asyncio.run(scenario())

    assert probe_result.reason == "timeout"
    assert response.message.content == "ok"


def test_hung_providers_are_probed_concurrently():
    upstreams = [Upstream(hang_seconds=30) for _ in range(4)]
    providers = {f"p{i}": provider_for(u, name=f"p{i}") for i, u in enumerate(upstreams)}
    registry = ModelRegistry(providers, [ModelRoute(f"m{i}", p, "m1") for i, p in enumerate(providers.values())])

    started = time.monotonic()
    report = check(registry, timeout=0.5)
    elapsed = time.monotonic() - started

    assert [c.reason for c in report.providers] == ["timeout"] * 4
    assert elapsed < 1.5  # about one timeout, not four
    assert all(u.cancelled for u in upstreams)


def test_one_failing_provider_does_not_stop_the_others():
    class Exploding(FakeProvider):
        async def list_models(self):
            raise RuntimeError("boom")

    hung = provider_for(Upstream(hang_seconds=30), name="hung")
    good = provider_for(Upstream(httpx.Response(200, json=models_body("m1"))), name="good")
    exploding = Exploding(name="exploding")
    registry = ModelRegistry(
        {"exploding": exploding, "good": good, "hung": hung},
        [ModelRoute("a", exploding, FAKE_MODEL), ModelRoute("b", good, "m1"), ModelRoute("c", hung, "m1")],
    )

    report = check(registry, timeout=0.3)

    assert [(c.name, c.status, c.reason) for c in report.providers] == [
        ("exploding", "unhealthy", "probe_error"),
        ("good", "healthy", None),
        ("hung", "unhealthy", "timeout"),
    ]
    assert {m.public_id: m.availability for m in report.models} == {"a": "unknown", "b": "listed", "c": "unknown"}


def test_timeout_must_be_positive():
    registry = ModelRegistry({"fake": FakeProvider()}, [])
    with pytest.raises(ValueError):
        asyncio.run(check_registry(registry, 0))


# --- Model availability is separate from provider health ---


def availability_registry(listing_upstream, *, extra_routes=()):
    groq = provider_for(listing_upstream, name="groq")
    down = provider_for(Upstream(raises=httpx.ConnectError("refused")), name="down")
    no_listing = provider_for(Upstream(httpx.Response(404, text="not found")), name="nolisting")
    registry = ModelRegistry(
        {"groq": groq, "down": down, "nolisting": no_listing},
        [
            ModelRoute("llama-3.3-70b", groq, "llama-3.3-70b-versatile"),
            ModelRoute("typo-model", groq, "llama-3.1-8b-instnt"),
            ModelRoute("local", down, "qwen2.5:7b"),
            ModelRoute("proxied", no_listing, "anything"),
        ],
        disabled=[ModelRoute("future-model", groq, "llama-4-preview"), ModelRoute("retired", groq, "llama-3.1-8b-instant")],
    )
    return registry, groq


def test_model_availability_is_derived_from_its_providers_listing():
    upstream = Upstream(httpx.Response(200, json=models_body("llama-3.3-70b-versatile", "llama-3.1-8b-instant")))
    registry, _ = availability_registry(upstream)

    report = check(registry)

    assert [(m.public_id, m.provider, m.enabled, m.availability) for m in report.models] == [
        ("future-model", "groq", False, "not_listed"),
        ("llama-3.3-70b", "groq", True, "listed"),
        ("local", "down", True, "unknown"),        # provider unreachable: nothing known
        ("proxied", "nolisting", True, "unknown"),  # provider can't list: nothing known
        ("retired", "groq", False, "listed"),       # disabled models are checked too
        ("typo-model", "groq", True, "not_listed"),  # healthy provider, misconfigured model
    ]
    statuses = {c.name: (c.status, c.reason) for c in report.providers}
    assert statuses == {
        "down": ("unhealthy", "connection_failed"),
        "groq": ("healthy", None),
        "nolisting": ("unverified", "listing_not_supported"),
    }


def test_matching_is_exact_so_aliases_are_reported_as_not_listed():
    # Ollama lists "qwen2.5:latest" but serves "qwen2.5" too: reported, never auto-corrected.
    ollama = provider_for(Upstream(httpx.Response(200, json=models_body("qwen2.5:latest"))), name="ollama")
    registry = ModelRegistry({"ollama": ollama}, [ModelRoute("qwen", ollama, "qwen2.5"), ModelRoute("Q", ollama, "QWEN2.5:LATEST")])

    report = check(registry)

    assert [m.availability for m in report.models] == ["not_listed", "not_listed"]


def test_problems_and_ok():
    healthy = ProviderCheck("groq", "healthy", None, 1.0)
    all_good = HealthReport((healthy,), (ModelCheck("m", "groq", "u", True, "listed"),))
    disabled_missing = HealthReport((healthy,), (ModelCheck("m", "groq", "u", False, "not_listed"),))
    enabled_missing = HealthReport((healthy,), (ModelCheck("m", "groq", "u", True, "not_listed"),))
    unverified = HealthReport((ProviderCheck("p", "unverified", "listing_not_supported", 1.0),), ())

    assert all_good.ok and all_good.problems == []
    assert disabled_missing.ok  # a disabled model can't affect traffic
    assert enabled_missing.problems == ["enabled model m is not_listed on provider groq"]
    assert unverified.problems == ["provider p is unverified (listing_not_supported)"]


# --- Probes never run inference and never leak ---


def test_probes_only_send_get_models_with_the_providers_key():
    upstream = Upstream(httpx.Response(200, json=models_body("m1")))
    provider = provider_for(upstream)
    registry = ModelRegistry({"upstream": provider}, [ModelRoute("a", provider, "m1"), ModelRoute("b", provider, "m2")])

    check(registry)

    [request] = upstream.requests  # one request per provider, however many models
    assert (request.method, request.url.path) == ("GET", "/v1/models")
    assert request.headers["Authorization"] == f"Bearer {SECRET}"


def test_reports_never_contain_upstream_details(caplog):
    message = f"upstream {BASE_URL} rejected key {SECRET} at 10.0.0.9"
    upstreams = [
        Upstream(httpx.Response(500, json={"error": {"message": message}})),
        Upstream(httpx.Response(401, json={"error": {"message": message}})),
        Upstream(raises=httpx.ConnectError(message)),
        Upstream(httpx.Response(200, text=message)),
    ]
    providers = {f"p{i}": provider_for(u, name=f"p{i}") for i, u in enumerate(upstreams)}
    registry = ModelRegistry(providers, [ModelRoute(f"m{i}", p, "x") for i, p in enumerate(providers.values())])

    with caplog.at_level(logging.INFO):
        report = check(registry)

    text = repr(report)
    for leaked in (SECRET, "internal-llm", "10.0.0.9", "rejected"):
        assert leaked not in text
    # Nothing from the application at INFO or above.
    # (httpx's own INFO request lines are outside the app; the CLI doesn't enable INFO logging.)
    assert [r for r in caplog.records if r.name.startswith("app.")] == []


def test_probe_failures_log_metadata_only_even_at_debug(caplog):
    message = f"upstream {BASE_URL} rejected key {SECRET} at 10.0.0.9"
    upstreams = [
        Upstream(httpx.Response(500, json={"error": {"message": message}})),
        Upstream(httpx.Response(401, json={"error": {"message": message}})),
        Upstream(raises=httpx.ConnectError(message)),
        Upstream(httpx.Response(200, text=message)),
    ]
    providers = {f"p{i}": provider_for(u, name=f"p{i}") for i, u in enumerate(upstreams)}
    registry = ModelRegistry(providers, [ModelRoute(f"m{i}", p, "x") for i, p in enumerate(providers.values())])

    with caplog.at_level(logging.DEBUG, logger="app"):
        check(registry)

    probe_lines = [r for r in caplog.records if r.name == "app.services.provider_health"]
    assert len(probe_lines) == 4
    assert "upstream status 401, reason authentication_failed" in caplog.text
    for record in probe_lines:
        assert not record.exc_info  # provider errors: no traceback, which would quote the message
    for leaked in (SECRET, "internal-llm", "10.0.0.9", "rejected key"):
        assert leaked not in caplog.text


def test_debug_logging_never_contains_the_key():
    """Debug logs carry the failure's metadata: neither the key nor the upstream message."""
    provider = provider_for(Upstream(httpx.Response(500, json={"error": {"message": "overloaded"}})))
    logger = logging.getLogger("app.services.provider_health")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        probe(provider)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    [record] = records
    assert "ProviderError (upstream status 500, reason server_error)" in record.getMessage()
    assert "overloaded" not in record.getMessage() and not record.exc_info
    assert SECRET not in record.getMessage()


# --- Compatibility of the provider interface ---


def test_health_keeps_its_existing_behaviour():
    healthy = provider_for(Upstream(httpx.Response(200, json=models_body("m1"))))
    down = provider_for(Upstream(raises=httpx.ConnectError("refused")))

    async def scenario():
        try:
            return await healthy.health(), await down.health()
        finally:
            await healthy.aclose()
            await down.aclose()

    ok, bad = asyncio.run(scenario())

    assert ok == HealthStatus(healthy=True)
    assert not bad.healthy and "could not reach provider" in bad.detail


def test_fake_provider_listing_fails_like_its_health_check():
    failing = FakeProvider(fail_with=ProviderUnavailableError("upstream down"))

    with pytest.raises(ProviderUnavailableError):
        asyncio.run(failing.list_models())
    assert not asyncio.run(failing.health()).healthy
    assert asyncio.run(FakeProvider().list_models()) == [ModelInfo(id=FAKE_MODEL, owned_by="fake")]


def test_default_registry_is_healthy():
    from app.providers.registry import default_registry

    report = check(default_registry())

    assert report.ok
    assert [(m.public_id, m.availability) for m in report.models] == [(FAKE_MODEL, "listed")]

