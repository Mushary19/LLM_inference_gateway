"""Quota reservation and settlement (app/services/quota.py) against the test PostgreSQL.

Single-request scenarios use the rolled-back test session; setup is committed (which only
releases a savepoint) because reserve() and settle() commit too. Concurrency scenarios use
real, separate sessions on committed rows, so PostgreSQL row locking is actually exercised.
"""

import threading
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import event, func, select, text, update
from sqlalchemy.exc import IntegrityError, OperationalError

from conftest import delete_committed_application

from app.database.connection import SessionLocal, engine
from app.database.models import QuotaReservation, ReservationStatus, TokenQuota, UsageRecord, UsageStatus
from app.providers.base import ChatMessage, Usage
from app.services import quota
from app.services.api_keys import create_api_key, create_application
from app.services.quota import (
    MESSAGE_OVERHEAD_TOKENS,
    REQUEST_OVERHEAD_TOKENS,
    InsufficientQuotaError,
    QuotaNotFoundError,
    estimate_request_tokens,
    expire_stale_reservations,
    quota_inconsistencies,
    reserve,
    restore_missing_quota,
    settle,
)


def make_application(db, name="quota-app", limit=1000, keys=1):
    application = create_application(db, name, token_limit=limit)
    key_ids = [create_api_key(db, application, f"key-{i}")[0].id for i in range(keys)]
    db.commit()
    return application.id, key_ids


def take(db, application_id, key_id, tokens):
    return reserve(
        db, application_id=application_id, api_key_id=key_id, request_id=uuid.uuid4(),
        tokens=tokens, provider="fake", model="fake-model",
    )


def quota_of(db, application_id) -> TokenQuota:
    db.expire_all()
    return db.get(TokenQuota, application_id)


def reservation_of(db, reservation_id) -> QuotaReservation:
    db.expire_all()
    return db.get(QuotaReservation, reservation_id)


def usage_rows(db, application_id) -> list[UsageRecord]:
    db.expire_all()
    return list(db.scalars(select(UsageRecord).where(UsageRecord.application_id == application_id)))


# --- Estimation ---


def test_estimate_counts_utf8_bytes_overheads_and_output_allowance():
    messages = (ChatMessage("system", "Be brief."), ChatMessage("user", "héllo"))  # 9 + 6 bytes

    estimate = estimate_request_tokens(messages, max_output_tokens=100)

    assert estimate == 9 + 6 + 2 * MESSAGE_OVERHEAD_TOKENS + REQUEST_OVERHEAD_TOKENS + 100


def test_estimate_grows_with_input_and_output():
    short = estimate_request_tokens((ChatMessage("user", "hi"),), 10)

    assert estimate_request_tokens((ChatMessage("user", "hi" * 50),), 10) > short
    assert estimate_request_tokens((ChatMessage("user", "hi"),), 500) > short
    assert estimate_request_tokens((ChatMessage("user", "日本語"),), 10) > estimate_request_tokens((ChatMessage("user", "abc"),), 10)


# --- Reservation ---


def test_successful_reservation_holds_tokens(db_session):
    app_id, [key] = make_application(db_session, limit=1000)

    held = take(db_session, app_id, key, 300)

    q = quota_of(db_session, app_id)
    assert (q.limit_tokens, q.consumed_tokens, q.reserved_tokens, q.available_tokens) == (1000, 0, 300, 700)
    row = reservation_of(db_session, held.reservation_id)
    assert row.status is ReservationStatus.RESERVED and row.reserved_tokens == 300
    assert row.charged_tokens is None and row.settled_at is None
    assert quota_inconsistencies(db_session, app_id) == []


def test_insufficient_quota_is_rejected_without_side_effects(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    take(db_session, app_id, key, 900)

    with pytest.raises(InsufficientQuotaError) as excinfo:
        take(db_session, app_id, key, 101)

    assert (excinfo.value.required_tokens, excinfo.value.available_tokens) == (101, 100)
    q = quota_of(db_session, app_id)
    assert (q.reserved_tokens, q.available_tokens) == (900, 100)
    assert db_session.scalar(select(func.count()).select_from(QuotaReservation)) == 1
    assert quota_inconsistencies(db_session, app_id) == []


def test_exact_quota_boundary(db_session):
    app_id, [key] = make_application(db_session, limit=1000)

    take(db_session, app_id, key, 1000)  # exactly the whole quota fits

    assert quota_of(db_session, app_id).available_tokens == 0
    with pytest.raises(InsufficientQuotaError):
        take(db_session, app_id, key, 1)


def test_application_without_a_quota_row_is_rejected(db_session):
    app_id, [key] = make_application(db_session)
    db_session.execute(text("DELETE FROM token_quotas WHERE application_id = :id"), {"id": app_id})
    db_session.commit()

    with pytest.raises(InsufficientQuotaError, match="no token quota"):
        take(db_session, app_id, key, 1)


def test_keys_of_one_application_share_its_quota(db_session):
    app_id, [key_a, key_b] = make_application(db_session, limit=500, keys=2)

    take(db_session, app_id, key_a, 300)
    take(db_session, app_id, key_b, 200)

    assert quota_of(db_session, app_id).available_tokens == 0
    with pytest.raises(InsufficientQuotaError):
        take(db_session, app_id, key_b, 1)


def test_invalid_reservation_size_is_rejected(db_session):
    app_id, [key] = make_application(db_session)

    with pytest.raises(ValueError):
        take(db_session, app_id, key, 0)


# --- Settlement ---


def test_reconciliation_charges_actual_usage_and_releases_the_rest(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 400)

    result = settle(
        db_session, held.reservation_id, usage=Usage(30, 70), succeeded=True,
        may_have_consumed_tokens=True, model="fake-model-v2", latency_ms=12.5,
    )

    assert (result.status, result.charged_tokens, result.already_settled) == (ReservationStatus.RECONCILED, 100, False)
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens, q.available_tokens) == (100, 0, 900)
    row = reservation_of(db_session, held.reservation_id)
    assert (row.actual_tokens, row.charged_tokens) == (100, 100) and row.settled_at is not None
    [record] = usage_rows(db_session, app_id)
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (30, 70, 100)
    assert record.usage_reported and record.charged_tokens == 100 and record.status is UsageStatus.SUCCEEDED
    assert (record.provider, record.model, record.latency_ms) == ("fake", "fake-model-v2", 12.5)
    assert record.request_id == held.request_id and record.api_key_id == key
    assert quota_inconsistencies(db_session, app_id) == []


def test_usage_above_the_reservation_is_charged_and_availability_never_goes_negative(db_session):
    app_id, [key] = make_application(db_session, limit=100)
    held = take(db_session, app_id, key, 80)

    settle(db_session, held.reservation_id, usage=Usage(60, 90), succeeded=True, may_have_consumed_tokens=True)

    q = quota_of(db_session, app_id)
    assert q.consumed_tokens == 150  # more than the limit: actual usage is always recorded
    assert q.available_tokens == 0
    with pytest.raises(InsufficientQuotaError):
        take(db_session, app_id, key, 1)
    assert quota_inconsistencies(db_session, app_id) == []


def test_success_without_reported_usage_charges_the_reservation(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 250)

    result = settle(db_session, held.reservation_id, usage=None, succeeded=True, may_have_consumed_tokens=True)

    assert (result.status, result.charged_tokens) == (ReservationStatus.RECONCILED, 250)
    [record] = usage_rows(db_session, app_id)
    assert not record.usage_reported
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (None, None, None)
    assert reservation_of(db_session, held.reservation_id).actual_tokens is None
    assert quota_of(db_session, app_id).consumed_tokens == 250


def test_failure_that_consumed_nothing_releases_the_reservation(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 250)

    result = settle(db_session, held.reservation_id, usage=None, succeeded=False, may_have_consumed_tokens=False)

    assert (result.status, result.charged_tokens) == (ReservationStatus.RELEASED, 0)
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens, q.available_tokens) == (0, 0, 1000)
    assert usage_rows(db_session, app_id) == []
    assert quota_inconsistencies(db_session, app_id) == []


def test_failure_that_may_have_consumed_tokens_charges_the_reservation(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 250)

    settle(db_session, held.reservation_id, usage=None, succeeded=False, may_have_consumed_tokens=True)

    [record] = usage_rows(db_session, app_id)
    assert record.status is UsageStatus.FAILED and not record.usage_reported and record.charged_tokens == 250
    assert quota_of(db_session, app_id).consumed_tokens == 250


def test_partial_failure_with_reported_usage_charges_actual_usage(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 250)

    settle(db_session, held.reservation_id, usage=Usage(20, 5), succeeded=False, may_have_consumed_tokens=True)

    [record] = usage_rows(db_session, app_id)
    assert record.status is UsageStatus.FAILED and record.usage_reported and record.charged_tokens == 25
    assert quota_of(db_session, app_id).consumed_tokens == 25


def test_repeated_settlement_never_charges_twice(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 400)
    settle(db_session, held.reservation_id, usage=Usage(30, 70), succeeded=True, may_have_consumed_tokens=True)

    again = settle(db_session, held.reservation_id, usage=Usage(300, 300), succeeded=True, may_have_consumed_tokens=True)
    expired = settle(
        db_session, held.reservation_id, usage=None, succeeded=False, may_have_consumed_tokens=True, expired=True
    )

    assert again.already_settled and again.charged_tokens == 100
    assert expired.already_settled and expired.status is ReservationStatus.RECONCILED
    assert quota_of(db_session, app_id).consumed_tokens == 100
    assert len(usage_rows(db_session, app_id)) == 1
    assert quota_inconsistencies(db_session, app_id) == []


def test_database_failure_during_settlement_leaves_consistent_state(db_session, monkeypatch):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 400)

    def failing_commit():
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    monkeypatch.setattr(db_session, "commit", failing_commit)
    with pytest.raises(OperationalError):
        settle(db_session, held.reservation_id, usage=Usage(30, 70), succeeded=True, may_have_consumed_tokens=True)
    monkeypatch.undo()

    # Rolled back as a whole: still reserved, nothing charged, no usage record.
    assert reservation_of(db_session, held.reservation_id).status is ReservationStatus.RESERVED
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (0, 400)
    assert usage_rows(db_session, app_id) == []
    assert quota_inconsistencies(db_session, app_id) == []
    # It can still be settled afterwards, exactly once.
    settle(db_session, held.reservation_id, usage=Usage(30, 70), succeeded=True, may_have_consumed_tokens=True)
    assert quota_of(db_session, app_id).consumed_tokens == 100


def test_settling_without_a_quota_row_fails_and_changes_nothing(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 400)
    # The quota row disappears between reservation and settlement (only possible by hand).
    db_session.execute(text("DELETE FROM token_quotas WHERE application_id = :id"), {"id": app_id})
    db_session.commit()

    with pytest.raises(QuotaNotFoundError, match=str(app_id)):
        settle(db_session, held.reservation_id, usage=Usage(30, 70), succeeded=True, may_have_consumed_tokens=True)

    # Nothing was released, charged or recorded, and the transaction was rolled back.
    assert not db_session.in_transaction()
    row = reservation_of(db_session, held.reservation_id)
    assert row.status is ReservationStatus.RESERVED and row.charged_tokens is None
    assert usage_rows(db_session, app_id) == []
    # Once the quota is restored (what `set-quota` does), the reservation can still be settled.
    restore_missing_quota(db_session, app_id, 1000)
    db_session.commit()
    settle(db_session, held.reservation_id, usage=Usage(30, 70), succeeded=True, may_have_consumed_tokens=True)
    assert quota_of(db_session, app_id).consumed_tokens == 100


# --- Expiry of reservations that were never settled ---


def test_stale_reservations_are_expired_and_charged(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    stale = take(db_session, app_id, key, 300)
    recent = take(db_session, app_id, key, 200)
    db_session.execute(
        update(QuotaReservation)
        .where(QuotaReservation.id == stale.reservation_id)
        .values(created_at=func.now() - timedelta(hours=2))
    )
    db_session.commit()

    expired = expire_stale_reservations(db_session, timedelta(hours=1))

    assert expired == [stale.request_id]
    row = reservation_of(db_session, stale.reservation_id)
    assert row.status is ReservationStatus.EXPIRED and row.charged_tokens == 300
    assert reservation_of(db_session, recent.reservation_id).status is ReservationStatus.RESERVED
    [record] = usage_rows(db_session, app_id)
    assert record.status is UsageStatus.EXPIRED and not record.usage_reported and record.charged_tokens == 300
    assert (record.provider, record.model) == ("fake", "fake-model")
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (300, 200)
    assert expire_stale_reservations(db_session, timedelta(hours=1)) == []  # idempotent
    assert quota_inconsistencies(db_session, app_id) == []


# --- Database constraints back the invariants ---


def test_consistency_check_detects_drifted_counters(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    take(db_session, app_id, key, 100)
    db_session.execute(update(TokenQuota).where(TokenQuota.application_id == app_id).values(reserved_tokens=99))
    db_session.commit()

    assert quota_inconsistencies(db_session, app_id) == ["reserved_tokens 99 != active reservations 100"]


def test_database_rejects_inconsistent_rows(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    held = take(db_session, app_id, key, 100)
    cases = [
        ("UPDATE token_quotas SET reserved_tokens = -1 WHERE application_id = :a", "ck_token_quotas_reserved_non_negative"),
        ("UPDATE token_quotas SET consumed_tokens = -1 WHERE application_id = :a", "ck_token_quotas_consumed_non_negative"),
        ("UPDATE quota_reservations SET status = 'released' WHERE id = :r", "ck_quota_reservations_settlement_consistent"),
        # Violates two checks; PostgreSQL reports whichever it evaluates first.
        ("UPDATE quota_reservations SET status = 'bogus' WHERE id = :r",
         "ck_quota_reservations_(status_valid|settlement_consistent)"),
        ("UPDATE quota_reservations SET reserved_tokens = 0 WHERE id = :r", "ck_quota_reservations_reserved_positive"),
    ]
    for statement, constraint in cases:
        with pytest.raises(IntegrityError, match=constraint):
            with db_session.begin_nested():
                db_session.execute(text(statement), {"a": app_id, "r": held.reservation_id})
    settle(db_session, held.reservation_id, usage=Usage(1, 1), succeeded=True, may_have_consumed_tokens=True)
    with pytest.raises(IntegrityError, match="ck_usage_records_token_counts_consistent"):
        with db_session.begin_nested():
            db_session.execute(text("UPDATE usage_records SET total_tokens = 99 WHERE application_id = :a"), {"a": app_id})
    with pytest.raises(IntegrityError, match="uq_quota_reservations_request_id"):
        with db_session.begin_nested():
            db_session.execute(
                text(
                    "INSERT INTO quota_reservations (application_id, api_key_id, request_id, provider, model, reserved_tokens)"
                    " VALUES (:a, :k, :q, 'fake', 'm', 1)"
                ),
                {"a": app_id, "k": key, "q": held.request_id},
            )


# --- Concurrency on real PostgreSQL (separate sessions, real row locks) ---


@pytest.fixture
def committed_application():
    name = f"quota-concurrency-{uuid.uuid4().hex[:8]}"

    def create(limit: int, keys: int = 1):
        with SessionLocal() as db:
            return make_application(db, name=name, limit=limit, keys=keys)

    yield create
    with SessionLocal() as db:
        delete_committed_application(db, name)


def run_in_threads(count: int, work) -> list:
    barrier = threading.Barrier(count)
    results: list = [None] * count

    def runner(i):
        barrier.wait()
        try:
            results[i] = work(i)
        except Exception as exc:  # recorded, asserted by the test
            results[i] = exc

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_reservations_never_exceed_the_quota(committed_application):
    app_id, keys = committed_application(limit=1000, keys=4)

    def attempt(i):
        with SessionLocal() as db:
            return take(db, app_id, keys[i % 4], 100)

    results = run_in_threads(25, attempt)

    succeeded = [r for r in results if isinstance(r, quota.Reservation)]
    rejected = [r for r in results if isinstance(r, InsufficientQuotaError)]
    assert len(succeeded) == 10 and len(rejected) == 15
    assert len(succeeded) + len(rejected) == 25  # no other errors
    with SessionLocal() as db:
        q = db.get(TokenQuota, app_id)
        assert (q.reserved_tokens, q.available_tokens) == (1000, 0)
        assert quota_inconsistencies(db, app_id) == []


def test_concurrent_settlement_of_one_reservation_charges_once(committed_application):
    app_id, [key] = committed_application(limit=1000)
    with SessionLocal() as db:
        held = take(db, app_id, key, 400)

    def attempt(i):
        with SessionLocal() as db:
            return settle(db, held.reservation_id, usage=Usage(10, 20 + i), succeeded=True, may_have_consumed_tokens=True)

    results = run_in_threads(10, attempt)

    assert all(isinstance(r, quota.Settlement) for r in results)
    assert [r.already_settled for r in results].count(False) == 1
    winner = next(r for r in results if not r.already_settled)
    with SessionLocal() as db:
        q = db.get(TokenQuota, app_id)
        assert (q.consumed_tokens, q.reserved_tokens) == (winner.charged_tokens, 0)
        assert db.scalar(select(func.count()).select_from(UsageRecord).where(UsageRecord.application_id == app_id)) == 1
        assert quota_inconsistencies(db, app_id) == []


def test_concurrent_reserve_and_settle_keep_counters_consistent(committed_application):
    app_id, keys = committed_application(limit=2000, keys=2)

    def attempt(i):
        with SessionLocal() as db:
            try:
                held = take(db, app_id, keys[i % 2], 150)
            except InsufficientQuotaError:
                return "rejected"
            usage = Usage(10, i) if i % 3 else None
            settle(db, held.reservation_id, usage=usage, succeeded=i % 4 != 0, may_have_consumed_tokens=i % 5 != 0)
            return "settled"

    results = run_in_threads(30, attempt)

    assert not [r for r in results if isinstance(r, Exception)]
    with SessionLocal() as db:
        q = db.get(TokenQuota, app_id)
        assert q.reserved_tokens == 0 and q.consumed_tokens >= 0 and q.available_tokens >= 0
        assert quota_inconsistencies(db, app_id) == []



# --- The consistency check reads one snapshot ---


def test_separate_queries_in_one_transaction_can_disagree(committed_application):
    """Why the diagnostic must be a single statement: under READ COMMITTED (PostgreSQL's
    default) each statement sees everything committed before it started, so two queries in
    one transaction can see different states if another session commits in between."""
    app_id, [key] = committed_application(limit=1000)
    with SessionLocal() as worker:
        held = take(worker, app_id, key, 300)

    with SessionLocal() as diagnostic, SessionLocal() as worker:
        counter = diagnostic.scalar(select(TokenQuota.reserved_tokens).where(TokenQuota.application_id == app_id))
        # Another request settles its reservation between the diagnostic's two queries.
        settle(worker, held.reservation_id, usage=Usage(10, 10), succeeded=True, may_have_consumed_tokens=True)
        active = diagnostic.scalar(
            select(func.coalesce(func.sum(QuotaReservation.reserved_tokens), 0)).where(
                QuotaReservation.application_id == app_id, QuotaReservation.status == ReservationStatus.RESERVED
            )
        )
        assert (counter, active) == (300, 0)  # a false "inconsistency" between two correct states
        assert quota_inconsistencies(diagnostic, app_id) == []


def test_consistency_check_is_a_single_statement(db_session):
    app_id, [key] = make_application(db_session, limit=1000)
    take(db_session, app_id, key, 100)
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        assert quota_inconsistencies(db_session, app_id) == []
    finally:
        event.remove(engine, "before_cursor_execute", record)

    # One read statement = one snapshot, whatever commits meanwhile. (The rolled-back test
    # session also issues a SAVEPOINT on first use; that's the fixture, not the diagnostic.)
    reads = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(reads) == 1


def test_consistency_check_reports_no_false_problems_under_concurrent_settlement(committed_application):
    app_id, keys = committed_application(limit=10_000_000, keys=3)
    stop = threading.Event()
    false_reports: list[list[str]] = []
    checks = 0

    def worker(i):
        with SessionLocal() as db:
            for n in range(40):
                held = take(db, app_id, keys[i], 100)
                settle(db, held.reservation_id, usage=Usage(n, i), succeeded=True, may_have_consumed_tokens=True)

    def diagnostic():
        nonlocal checks
        with SessionLocal() as db:
            while not stop.is_set():
                problems = quota_inconsistencies(db, app_id)
                db.rollback()
                checks += 1
                if problems:
                    false_reports.append(problems)

    checker = threading.Thread(target=diagnostic)
    workers = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    checker.start()
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    stop.set()
    checker.join()

    assert checks > 10  # the diagnostic really ran alongside the workers
    assert false_reports == []
    with SessionLocal() as db:
        assert quota_inconsistencies(db, app_id) == []



# --- Restoring a deleted quota row ---


def delete_quota_row(db, application_id):
    db.execute(text("DELETE FROM token_quotas WHERE application_id = :id"), {"id": application_id})
    db.commit()


def history(db, app_id, key):
    """A reservation history with every status: 2 still reserved, 3 settled (one released)."""
    outstanding = [take(db, app_id, key, 2000), take(db, app_id, key, 700)]
    reconciled = take(db, app_id, key, 500)
    settle(db, reconciled.reservation_id, usage=Usage(40, 60), succeeded=True, may_have_consumed_tokens=True)
    released = take(db, app_id, key, 300)
    settle(db, released.reservation_id, usage=None, succeeded=False, may_have_consumed_tokens=False)
    unreported = take(db, app_id, key, 250)
    settle(db, unreported.reservation_id, usage=None, succeeded=True, may_have_consumed_tokens=True)
    return outstanding  # charged so far: 100 + 0 + 250 = 350; still reserved: 2700


def test_restoring_rebuilds_counters_from_outstanding_reservations(db_session):
    app_id, [key] = make_application(db_session, limit=100_000)
    take(db_session, app_id, key, 2000)
    delete_quota_row(db_session, app_id)

    restore_missing_quota(db_session, app_id, 100_000)
    db_session.commit()

    q = quota_of(db_session, app_id)
    assert (q.limit_tokens, q.consumed_tokens, q.reserved_tokens, q.available_tokens) == (100_000, 0, 2000, 98_000)
    assert quota_inconsistencies(db_session, app_id) == []


def test_restoring_rebuilds_counters_from_settled_reservations(db_session):
    app_id, [key] = make_application(db_session, limit=100_000)
    held = take(db_session, app_id, key, 500)
    settle(db_session, held.reservation_id, usage=Usage(40, 60), succeeded=True, may_have_consumed_tokens=True)
    delete_quota_row(db_session, app_id)

    restore_missing_quota(db_session, app_id, 100_000)
    db_session.commit()

    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (100, 0)
    assert len(usage_rows(db_session, app_id)) == 1  # history untouched
    assert quota_inconsistencies(db_session, app_id) == []


def test_restoring_with_mixed_history_then_settling_the_outstanding_reservations(db_session):
    app_id, [key] = make_application(db_session, limit=100_000)
    outstanding = history(db_session, app_id, key)
    reservations_before = db_session.scalar(select(func.count()).select_from(QuotaReservation))
    usage_before = len(usage_rows(db_session, app_id))
    delete_quota_row(db_session, app_id)

    restore_missing_quota(db_session, app_id, 100_000)
    db_session.commit()

    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (350, 2700)
    assert quota_inconsistencies(db_session, app_id) == []
    # The outstanding reservations settle normally against the restored counters.
    settle(db_session, outstanding[0].reservation_id, usage=Usage(900, 100), succeeded=True, may_have_consumed_tokens=True)
    settle(db_session, outstanding[1].reservation_id, usage=None, succeeded=False, may_have_consumed_tokens=False)
    q = quota_of(db_session, app_id)
    assert (q.consumed_tokens, q.reserved_tokens) == (1350, 0)
    assert quota_inconsistencies(db_session, app_id) == []
    # No reservation was deleted or recreated; one new usage record (the released one has none).
    assert db_session.scalar(select(func.count()).select_from(QuotaReservation)) == reservations_before
    assert len(usage_rows(db_session, app_id)) == usage_before + 1


def test_restoring_an_existing_quota_is_refused(db_session):
    app_id, _ = make_application(db_session, limit=1000)

    with pytest.raises(IntegrityError, match="token_quotas_pkey"):
        restore_missing_quota(db_session, app_id, 5000)
    db_session.rollback()

    assert quota_of(db_session, app_id).limit_tokens == 1000


def test_concurrent_restorations_create_exactly_one_correct_row(committed_application):
    app_id, [key] = committed_application(limit=100_000)
    with SessionLocal() as db:
        history(db, app_id, key)
        delete_quota_row(db, app_id)

    def attempt(i):
        with SessionLocal() as db:
            restore_missing_quota(db, app_id, 100_000 + i)
            db.commit()
            return "restored"

    results = run_in_threads(6, attempt)

    assert results.count("restored") == 1
    losers = [r for r in results if r != "restored"]
    assert len(losers) == 5 and all(isinstance(r, IntegrityError) and "token_quotas_pkey" in str(r) for r in losers)
    with SessionLocal() as db:
        q = db.get(TokenQuota, app_id)
        assert (q.consumed_tokens, q.reserved_tokens) == (350, 2700)
        assert quota_inconsistencies(db, app_id) == []


def test_reservations_racing_a_restoration_stay_consistent(committed_application):
    """Requests keep arriving while the quota is missing and while it's being restored."""
    app_id, keys = committed_application(limit=100_000, keys=2)
    with SessionLocal() as db:
        history(db, app_id, keys[0])
        delete_quota_row(db, app_id)
    restored = threading.Event()
    outcomes: list[str] = []
    lock = threading.Lock()

    def request_loop(i):
        with SessionLocal() as db:
            for _ in range(30):
                try:
                    held = take(db, app_id, keys[i % 2], 100)
                except InsufficientQuotaError:  # no quota row yet
                    outcome = "rejected"
                else:
                    settle(db, held.reservation_id, usage=Usage(10, 10), succeeded=True, may_have_consumed_tokens=True)
                    outcome = "settled"
                with lock:
                    outcomes.append(outcome)
                if restored.is_set() and outcome == "settled":
                    break

    def restorer():
        with SessionLocal() as db:
            restore_missing_quota(db, app_id, 100_000)
            db.commit()
        restored.set()

    workers = [threading.Thread(target=request_loop, args=(i,)) for i in range(4)]
    for t in workers:
        t.start()
    restoring = threading.Thread(target=restorer)
    restoring.start()
    for t in [*workers, restoring]:
        t.join()

    assert "settled" in outcomes  # requests did run after the restoration
    with SessionLocal() as db:
        q = db.get(TokenQuota, app_id)
        settled = outcomes.count("settled")
        assert (q.consumed_tokens, q.reserved_tokens) == (350 + 20 * settled, 2700)
        assert quota_inconsistencies(db, app_id) == []
