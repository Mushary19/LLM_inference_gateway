"""Quota reservation in POST /v1/chat/completions, and the usage endpoints."""

import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError

from conftest import delete_committed_application

from app.api.chat import get_chat_provider
from app.core.config import settings
from app.database.connection import SessionLocal
from app.database.models import QuotaReservation, ReservationStatus, TokenQuota, UsageRecord, UsageStatus
from app.main import app
from app.providers.base import (
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ModelNotFoundError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.providers.fake import FAKE_MODEL
from app.services import quota as quota_service
from app.services.api_keys import create_api_key, create_application
from app.services.quota import MESSAGE_OVERHEAD_TOKENS, REQUEST_OVERHEAD_TOKENS, quota_inconsistencies

CHAT = "/v1/chat/completions"
BODY = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
ESTIMATE = len("hi") + MESSAGE_OVERHEAD_TOKENS + REQUEST_OVERHEAD_TOKENS + 10  # 92
INTERNAL_IDS = {"id", "application_id", "api_key_id", "reservation_id"}


class ScriptedProvider:
    """Returns a fixed usage (or none), or raises; records calls; can block on a gate."""

    name = "scripted"

    def __init__(self, usage=Usage(3, 4), error=None, gate: threading.Event | None = None) -> None:
        self.usage = usage
        self.error = error
        self.gate = gate
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        if self.gate is not None:
            await asyncio.to_thread(self.gate.wait, 10)
        if self.error is not None:
            raise self.error
        return ChatResponse(request.model, ChatMessage("assistant", "ok"), "stop", self.usage)

    async def list_models(self):
        return [ModelInfo(id=FAKE_MODEL, owned_by="scripted")]


@pytest.fixture
def use_provider():
    def install(provider):
        app.dependency_overrides[get_chat_provider] = lambda: provider
        return provider

    yield install
    app.dependency_overrides.pop(get_chat_provider, None)


def set_limit(db, application_id, limit):
    db.execute(update(TokenQuota).where(TokenQuota.application_id == application_id).values(limit_tokens=limit))
    db.commit()


def quota_of(db, application_id) -> TokenQuota:
    db.expire_all()
    return db.get(TokenQuota, application_id)


def reservations(db, application_id) -> list[QuotaReservation]:
    db.expire_all()
    return list(db.scalars(select(QuotaReservation).where(QuotaReservation.application_id == application_id)))


def usage_rows(db, application_id) -> list[UsageRecord]:
    db.expire_all()
    return list(db.scalars(select(UsageRecord).where(UsageRecord.application_id == application_id)))


def request_id_of(response) -> uuid.UUID:
    return uuid.UUID(hex=response.json()["id"].removeprefix("chatcmpl-"))


# --- Chat completions reserve and settle quota ---


def test_completion_is_reserved_reconciled_and_recorded(client, db_session, api_key, auth_headers):
    app_id = api_key.record.application_id

    response = client.post(CHAT, json=BODY, headers=auth_headers)

    assert response.status_code == 200
    usage = response.json()["usage"]  # FakeProvider: "hi" -> "Fake response to: hi"
    assert usage == {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5}
    [reservation] = reservations(db_session, app_id)
    assert reservation.request_id == request_id_of(response)  # the response ID is the request ID
    assert reservation.status is ReservationStatus.RECONCILED
    assert (reservation.reserved_tokens, reservation.actual_tokens, reservation.charged_tokens) == (ESTIMATE, 5, 5)
    [record] = usage_rows(db_session, app_id)
    assert (record.provider, record.model, record.status) == ("fake", FAKE_MODEL, UsageStatus.SUCCEEDED)
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens, record.charged_tokens) == (1, 4, 5, 5)
    assert record.api_key_id == api_key.record.id and record.latency_ms >= 0
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (5, 0)
    assert quota_inconsistencies(db_session, app_id) == []


def test_insufficient_quota_returns_429_without_calling_the_provider(client, db_session, api_key, auth_headers, use_provider):
    provider = use_provider(ScriptedProvider())
    app_id = api_key.record.application_id
    set_limit(db_session, app_id, ESTIMATE - 1)

    response = client.post(CHAT, json=BODY, headers=auth_headers)

    assert response.status_code == 429
    assert response.json() == {
        "detail": {
            "code": "insufficient_quota",
            "message": "Not enough token quota left for this request",
            "required_tokens": ESTIMATE,
            "available_tokens": ESTIMATE - 1,
        }
    }
    assert "Retry-After" not in response.headers  # unlike a rate limit, a quota doesn't refill
    assert provider.calls == 0
    assert reservations(db_session, app_id) == [] and usage_rows(db_session, app_id) == []
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (0, 0)


def test_exact_quota_boundary_over_http(client, db_session, api_key, auth_headers, use_provider):
    use_provider(ScriptedProvider(usage=Usage(2, 3)))
    app_id = api_key.record.application_id
    set_limit(db_session, app_id, ESTIMATE)

    first = client.post(CHAT, json=BODY, headers=auth_headers)  # reserves exactly the whole quota
    second = client.post(CHAT, json=BODY, headers=auth_headers)  # 87 left < 92 needed

    assert first.status_code == 200 and second.status_code == 429
    assert second.json()["detail"]["available_tokens"] == ESTIMATE - 5


def test_request_without_max_tokens_reserves_the_generation_limit(client, db_session, api_key, auth_headers):
    body = {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "hi"}]}

    assert client.post(CHAT, json=body, headers=auth_headers).status_code == 200

    [reservation] = reservations(db_session, api_key.record.application_id)
    assert reservation.reserved_tokens == ESTIMATE - 10 + settings.max_generation_tokens


def test_keys_of_one_application_share_its_quota_over_http(client, db_session, api_key, auth_headers, use_provider):
    use_provider(ScriptedProvider(usage=Usage(0, 0)))
    app_id = api_key.record.application_id
    application = db_session.get(type(api_key.record.application), app_id)
    _, second_key = create_api_key(db_session, application, "second")
    db_session.commit()
    set_limit(db_session, app_id, ESTIMATE * 2)
    second_headers = {"Authorization": f"Bearer {second_key}"}

    assert client.post(CHAT, json=BODY, headers=auth_headers).status_code == 200
    assert client.post(CHAT, json=BODY, headers=second_headers).status_code == 200
    q = quota_of(db_session, app_id)
    assert q.consumed_tokens == 0 and q.available_tokens == ESTIMATE * 2  # zero actual usage, all released


@pytest.mark.parametrize(
    "error, http_status, status, charged, usage_record",
    [
        (ModelNotFoundError("gone"), 404, ReservationStatus.RELEASED, 0, None),
        (ProviderUnavailableError("refused", may_have_consumed_tokens=False), 503, ReservationStatus.RELEASED, 0, None),
        (ProviderTimeoutError("read timeout"), 504, ReservationStatus.RECONCILED, ESTIMATE, (False, None)),
        (ProviderUnavailableError("dropped", usage=Usage(5, 3)), 503, ReservationStatus.RECONCILED, 8, (True, 8)),
    ],
    ids=["no-usage-released", "never-sent-released", "may-have-consumed-charged", "partial-with-usage"],
)
def test_provider_failures_settle_by_policy(
    client, db_session, api_key, auth_headers, use_provider, error, http_status, status, charged, usage_record
):
    use_provider(ScriptedProvider(error=error))
    app_id = api_key.record.application_id

    response = client.post(CHAT, json=BODY, headers=auth_headers)

    assert response.status_code == http_status  # existing provider error mapping is unchanged
    [reservation] = reservations(db_session, app_id)
    assert (reservation.status, reservation.charged_tokens) == (status, charged)
    records = usage_rows(db_session, app_id)
    if usage_record is None:
        assert records == []
    else:
        [record] = records
        assert record.status is UsageStatus.FAILED
        assert (record.usage_reported, record.total_tokens) == usage_record
        assert record.charged_tokens == charged
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (charged, 0)
    assert quota_inconsistencies(db_session, app_id) == []


def test_missing_provider_usage_is_charged_conservatively_and_reported_as_null(
    client, db_session, api_key, auth_headers, use_provider
):
    use_provider(ScriptedProvider(usage=None))
    app_id = api_key.record.application_id

    response = client.post(CHAT, json=BODY, headers=auth_headers)

    assert response.status_code == 200 and response.json()["usage"] is None
    [record] = usage_rows(db_session, app_id)
    assert not record.usage_reported and record.total_tokens is None and record.charged_tokens == ESTIMATE
    assert quota_of(db_session, app_id).consumed_tokens == ESTIMATE


def test_usage_above_the_reservation_is_charged(client, db_session, api_key, auth_headers, use_provider):
    use_provider(ScriptedProvider(usage=Usage(500, 500)))
    app_id = api_key.record.application_id
    set_limit(db_session, app_id, 200)

    assert client.post(CHAT, json=BODY, headers=auth_headers).status_code == 200

    q = quota_of(db_session, app_id)
    assert q.consumed_tokens == 1000 and q.available_tokens == 0
    assert client.get("/v1/usage/quota", headers=auth_headers).json()["available_tokens"] == 0
    assert client.post(CHAT, json=BODY, headers=auth_headers).status_code == 429


def test_database_failure_during_settlement_keeps_the_response_and_the_reservation(
    client, db_session, api_key, auth_headers, use_provider, monkeypatch, caplog
):
    use_provider(ScriptedProvider(usage=Usage(3, 4)))
    app_id = api_key.record.application_id

    def failing_settle(*args, **kwargs):
        raise OperationalError("UPDATE", {}, Exception("connection lost"))

    monkeypatch.setattr(quota_service, "settle", failing_settle)
    response = client.post(CHAT, json=BODY, headers=auth_headers)
    monkeypatch.undo()

    assert response.status_code == 200  # the provider already did the work
    assert "stays reserved until expired" in caplog.text
    [reservation] = reservations(db_session, app_id)
    assert reservation.status is ReservationStatus.RESERVED
    assert quota_of(db_session, app_id).reserved_tokens == ESTIMATE
    assert quota_inconsistencies(db_session, app_id) == []

    # Recovery: expiring it charges the reservation exactly once.
    expired = quota_service.expire_stale_reservations(db_session, timedelta(0))
    assert expired == [request_id_of(response)]
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (ESTIMATE, 0)
    assert [r.status for r in usage_rows(db_session, app_id)] == [UsageStatus.EXPIRED]
    assert quota_inconsistencies(db_session, app_id) == []


def test_missing_quota_row_at_settlement_keeps_the_response_and_the_reservation(
    client, db_session, api_key, auth_headers, use_provider, caplog
):
    app_id = api_key.record.application_id

    class QuotaDeletingProvider(ScriptedProvider):
        async def chat(self, request):
            # The quota row disappears while the provider runs (only possible by hand).
            db_session.execute(TokenQuota.__table__.delete().where(TokenQuota.application_id == app_id))
            db_session.commit()
            return await super().chat(request)

    use_provider(QuotaDeletingProvider())

    response = client.post(CHAT, json=BODY, headers=auth_headers)

    assert response.status_code == 200  # the client still gets the completed response
    assert "stays reserved until expired" in caplog.text
    [reservation] = reservations(db_session, app_id)
    assert reservation.status is ReservationStatus.RESERVED and reservation.charged_tokens is None
    assert usage_rows(db_session, app_id) == []


def test_database_failure_during_reservation_returns_503_without_calling_the_provider(
    client, db_session, api_key, auth_headers, use_provider, monkeypatch
):
    provider = use_provider(ScriptedProvider())

    def failing_reserve(*args, **kwargs):
        raise OperationalError("SELECT", {}, Exception("connection lost"))

    monkeypatch.setattr(quota_service, "reserve", failing_reserve)
    response = client.post(CHAT, json=BODY, headers=auth_headers)

    assert response.status_code == 503
    assert response.json() == {"detail": "Quota accounting is temporarily unavailable"}
    assert provider.calls == 0


def test_model_listing_does_not_use_quota(client, db_session, api_key, auth_headers):
    assert client.get("/v1/models", headers=auth_headers).status_code == 200

    assert reservations(db_session, api_key.record.application_id) == []


# --- Concurrency over HTTP (real per-request sessions) ---


@pytest.fixture
def committed_key():
    name = f"usage-concurrency-{uuid.uuid4().hex[:8]}"
    with SessionLocal() as db:
        application = create_application(db, name, token_limit=ESTIMATE * 3)
        _, plaintext = create_api_key(db, application, "concurrency")
        application_id = application.id
        db.commit()
    yield plaintext, application_id
    with SessionLocal() as db:
        delete_committed_application(db, name)


def test_concurrent_requests_cannot_exceed_the_quota(committed_key, use_provider):
    plaintext, app_id = committed_key
    gate = threading.Event()
    provider = use_provider(ScriptedProvider(usage=Usage(2, 3), gate=gate))
    headers = {"Authorization": f"Bearer {plaintext}"}
    statuses: list[int] = []
    lock = threading.Lock()

    with TestClient(app) as client:
        barrier = threading.Barrier(10)

        def call():
            barrier.wait()
            status = client.post(CHAT, json=BODY, headers=headers).status_code
            with lock:
                statuses.append(status)

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        # Admitted requests wait inside the provider, holding their reservations, until all
        # the others have been answered; only then are they released.
        deadline = time.monotonic() + 10
        while not (provider.calls == 3 and len(statuses) == 7) and time.monotonic() < deadline:
            time.sleep(0.01)
        gate.set()
        for t in threads:
            t.join()

    assert sorted(statuses) == [200] * 3 + [429] * 7
    assert provider.calls == 3
    with SessionLocal() as db:
        q = db.get(TokenQuota, app_id)
        assert (q.consumed_tokens, q.reserved_tokens) == (15, 0)
        assert quota_inconsistencies(db, app_id) == []


# --- GET /v1/usage ---


def record_usage(db, application_id, key_id, *, model="m1", provider="p1", usage=Usage(1, 2), at=None):
    """Create a settled usage record through the real service, optionally back-dated."""
    held = quota_service.reserve(
        db, application_id=application_id, api_key_id=key_id, request_id=uuid.uuid4(),
        tokens=100, provider=provider, model=model,
    )
    quota_service.settle(db, held.reservation_id, usage=usage, succeeded=True, may_have_consumed_tokens=True)
    if at is not None:
        db.execute(update(UsageRecord).where(UsageRecord.request_id == held.request_id).values(created_at=at))
        db.commit()
    return held.request_id


@pytest.mark.parametrize("url", ["/v1/usage", "/v1/usage/quota"])
def test_usage_endpoints_require_authentication(client, url):
    response = client.get(url)

    assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"


def test_usage_report_lists_this_applications_usage(client, db_session, api_key, auth_headers):
    first = client.post(CHAT, json=BODY, headers=auth_headers)
    second = client.post(CHAT, json=BODY, headers=auth_headers)

    response = client.get("/v1/usage", headers=auth_headers)

    assert response.status_code == 200
    report = response.json()
    assert report["object"] == "usage"
    assert report["summary"] == {
        "requests": 2,
        "requests_without_reported_usage": 0,
        "prompt_tokens": 2,
        "completion_tokens": 8,
        "total_tokens": 10,
        "charged_tokens": 10,
    }
    # Both rows share one created_at here (the rolled-back test session is one transaction),
    # so order is checked in test_usage_pagination with distinct timestamps instead.
    assert {r["request_id"] for r in report["data"]} == {first.json()["id"], second.json()["id"]}
    item = report["data"][0]
    assert set(item) & INTERNAL_IDS == set()
    assert item["key_prefix"] == api_key.record.key_prefix
    assert (item["provider"], item["model"], item["status"], item["usage_reported"]) == ("fake", FAKE_MODEL, "succeeded", True)
    assert report["pagination"] == {"limit": 50, "offset": 0, "has_more": False}
    assert response.headers["X-RateLimit-Limit"]  # usage endpoints are rate-limited too


def test_usage_is_isolated_per_application(client, db_session, api_key, auth_headers):
    other = create_application(db_session, "other-app")
    other_key, other_plaintext = create_api_key(db_session, other, "other")
    db_session.commit()
    record_usage(db_session, other.id, other_key.id, usage=Usage(100, 100))
    client.post(CHAT, json=BODY, headers=auth_headers)

    mine = client.get("/v1/usage", headers=auth_headers).json()
    theirs = client.get("/v1/usage", headers={"Authorization": f"Bearer {other_plaintext}"}).json()

    assert (mine["summary"]["requests"], mine["summary"]["total_tokens"]) == (1, 5)
    assert (theirs["summary"]["requests"], theirs["summary"]["total_tokens"]) == (1, 200)
    assert client.get("/v1/usage/quota", headers=auth_headers).json()["consumed_tokens"] == 5


def test_usage_filters(client, db_session, api_key, auth_headers):
    app_id, key_id = api_key.record.application_id, api_key.record.id
    day = datetime(2026, 3, 1, tzinfo=timezone.utc)
    record_usage(db_session, app_id, key_id, model="m1", provider="p1", at=day)
    record_usage(db_session, app_id, key_id, model="m2", provider="p1", at=day + timedelta(days=1))
    record_usage(db_session, app_id, key_id, model="m1", provider="p2", at=day + timedelta(days=2))

    def requests(**params):
        response = client.get("/v1/usage", headers=auth_headers, params=params)
        assert response.status_code == 200, response.text
        return response.json()["summary"]["requests"]

    assert requests() == 3
    assert requests(model="m1") == 2
    assert requests(provider="p1") == 2
    assert requests(model="m1", provider="p2") == 1
    assert requests(start=(day + timedelta(days=1)).isoformat()) == 2  # inclusive start
    assert requests(end=(day + timedelta(days=1)).isoformat()) == 1  # exclusive end
    assert requests(start=day.isoformat(), end=(day + timedelta(days=2)).isoformat()) == 2
    assert requests(start="2026-03-02T00:00:00") == 2  # no offset: taken as UTC
    assert requests(model="nope") == 0


def test_invalid_usage_filters_are_rejected(client, auth_headers):
    for params in (
        {"start": "2026-03-02T00:00:00Z", "end": "2026-03-01T00:00:00Z"},
        {"start": "2026-03-01T00:00:00Z", "end": "2026-03-01T00:00:00Z"},
        {"start": "not-a-date"},
        {"limit": 0},
        {"limit": 201},
        {"offset": -1},
    ):
        assert client.get("/v1/usage", headers=auth_headers, params=params).status_code == 422, params


def test_usage_pagination(client, db_session, api_key, auth_headers):
    app_id, key_id = api_key.record.application_id, api_key.record.id
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    ids = [record_usage(db_session, app_id, key_id, at=base + timedelta(minutes=i)) for i in range(5)]
    expected = [f"chatcmpl-{i.hex}" for i in reversed(ids)]  # newest first

    pages = [client.get("/v1/usage", headers=auth_headers, params={"limit": 2, "offset": o}).json() for o in (0, 2, 4)]

    assert [r["request_id"] for page in pages for r in page["data"]] == expected
    assert [page["pagination"]["has_more"] for page in pages] == [True, True, False]
    assert all(page["summary"]["requests"] == 5 for page in pages)  # totals cover every page


def test_unreported_usage_is_counted_separately_in_the_summary(client, db_session, api_key, auth_headers, use_provider):
    use_provider(ScriptedProvider(usage=None))
    client.post(CHAT, json=BODY, headers=auth_headers)

    summary = client.get("/v1/usage", headers=auth_headers).json()["summary"]

    assert summary["requests_without_reported_usage"] == 1
    assert summary["total_tokens"] == 0  # nothing invented
    assert summary["charged_tokens"] == ESTIMATE


# --- GET /v1/usage/quota ---


def test_quota_report(client, db_session, api_key, auth_headers):
    set_limit(db_session, api_key.record.application_id, 1000)
    client.post(CHAT, json=BODY, headers=auth_headers)

    response = client.get("/v1/usage/quota", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {
        "object": "quota",
        "limit_tokens": 1000,
        "consumed_tokens": 5,
        "reserved_tokens": 0,
        "available_tokens": 995,
    }


def test_quota_report_without_a_quota_row(client, db_session, api_key, auth_headers):
    db_session.execute(
        TokenQuota.__table__.delete().where(TokenQuota.application_id == api_key.record.application_id)
    )
    db_session.commit()

    response = client.get("/v1/usage/quota", headers=auth_headers)

    assert response.status_code == 404
    assert client.post(CHAT, json=BODY, headers=auth_headers).json()["detail"]["code"] == "insufficient_quota"


def test_new_applications_get_the_default_quota(db_session):
    application = create_application(db_session, "defaulted")
    db_session.commit()

    q = db_session.get(TokenQuota, application.id)
    assert (q.limit_tokens, q.consumed_tokens, q.reserved_tokens) == (settings.default_token_quota, 0, 0)
    assert db_session.scalar(select(func.count()).select_from(TokenQuota)) == 1
