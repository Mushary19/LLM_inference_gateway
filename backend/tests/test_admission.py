import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import SessionLocal
from app.database.models import InferenceRequest, RequestStatus
from app.main import app
from app.services.admission import AdmissionController, AdmissionDecision

PAYLOAD = {"prompt": "Explain KV caching", "max_new_tokens": 100, "priority": 2}


def _seed(db_session: Session, count: int, status: RequestStatus = RequestStatus.QUEUED) -> None:
    db_session.add_all(
        InferenceRequest(prompt=f"seed {i}", max_new_tokens=1, input_tokens=2, status=status)
        for i in range(count)
    )
    # Releases the test savepoint so the endpoint's rollback on rejection cannot undo the seed rows.
    db_session.commit()


def _count(db_session: Session, status: RequestStatus | None = None) -> int:
    query = select(func.count()).select_from(InferenceRequest)
    if status is not None:
        query = query.where(InferenceRequest.status == status)
    return db_session.scalar(query)


# --- AdmissionController unit tests (no database) ---


@pytest.fixture
def controller() -> AdmissionController:
    return AdmissionController(max_queued_requests=5, max_active_requests=3)


@pytest.mark.parametrize("queued", [0, 4])
def test_controller_accepts_below_capacity(controller, queued):
    result = controller.decide(queued)

    assert result.decision is AdmissionDecision.ACCEPT
    assert result.accepted and result.reason is None


@pytest.mark.parametrize("queued", [5, 6, 100])
def test_controller_rejects_at_or_above_capacity(controller, queued):
    result = controller.decide(queued)

    assert result.decision is AdmissionDecision.REJECT
    assert not result.accepted
    assert result.reason.startswith("Queue capacity reached")


def test_controller_uses_configured_limits():
    controller = AdmissionController(max_queued_requests=1, max_active_requests=3)

    assert controller.decide(0).accepted
    assert not controller.decide(1).accepted


def test_default_limits_from_settings():
    assert settings.max_active_requests == 3
    assert settings.max_queued_requests == 5


# --- AdmissionController.evaluate against PostgreSQL ---


def test_evaluate_counts_only_queued_requests(controller, db_session):
    for status in RequestStatus:
        if status is not RequestStatus.QUEUED:
            _seed(db_session, 5, status)
    _seed(db_session, 4)

    assert controller.evaluate(db_session).accepted

    _seed(db_session, 1)
    assert not controller.evaluate(db_session).accepted


# --- POST /generate with admission control ---


def test_accepts_with_empty_queue(client, db_session):
    response = client.post("/generate", json=PAYLOAD)

    assert response.status_code == 202
    assert response.json()["status"] == "QUEUED"
    assert _count(db_session, RequestStatus.QUEUED) == 1


def test_accepts_up_to_queue_capacity(client, db_session):
    _seed(db_session, 4)

    response = client.post("/generate", json=PAYLOAD)

    assert response.status_code == 202
    assert _count(db_session, RequestStatus.QUEUED) == 5


def test_rejects_at_queue_capacity_without_persisting(client, db_session):
    _seed(db_session, 5)

    response = client.post("/generate", json=PAYLOAD)

    assert response.status_code == 503
    assert response.json() == {"detail": "Inference system is temporarily at capacity"}
    assert _count(db_session, RequestStatus.QUEUED) == 5
    assert _count(db_session) == 5
    prompt_rows = select(func.count()).where(InferenceRequest.prompt == PAYLOAD["prompt"])
    assert db_session.scalar(prompt_rows) == 0


def test_rejection_does_not_consume_capacity(client, db_session):
    """A rejection must not consume capacity: the queue stays full, so the next request is also rejected."""
    _seed(db_session, 5)

    assert client.post("/generate", json=PAYLOAD).status_code == 503
    assert client.post("/generate", json=PAYLOAD).status_code == 503
    assert _count(db_session) == 5


def test_non_queued_statuses_do_not_consume_queue_capacity(client, db_session):
    non_queued = [s for s in RequestStatus if s is not RequestStatus.QUEUED]
    assert {s.value for s in non_queued} == {"RUNNING", "COMPLETED", "FAILED", "REJECTED", "CANCELLED"}
    for status in non_queued:
        _seed(db_session, 5, status)

    for _ in range(5):
        assert client.post("/generate", json=PAYLOAD).status_code == 202
    assert client.post("/generate", json=PAYLOAD).status_code == 503

    assert _count(db_session, RequestStatus.QUEUED) == 5
    assert _count(db_session) == 5 * len(non_queued) + 5


def test_validation_runs_before_admission(client, db_session):
    _seed(db_session, 5)

    response = client.post("/generate", json={"prompt": "   ", "max_new_tokens": 50})

    assert response.status_code == 422


# --- Concurrency (real commits, cleaned up afterwards) ---


def test_concurrent_requests_never_exceed_queue_capacity():
    """Fire more simultaneous requests than capacity; exactly max_queued_requests must be accepted.

    Uses real committed transactions (not the rollback fixture) so the requests genuinely race.
    """
    marker = f"concurrency-test-{uuid.uuid4()}"
    with SessionLocal() as db:
        existing = db.scalar(
            select(func.count()).where(InferenceRequest.status == RequestStatus.QUEUED)
        )
    if existing:
        pytest.skip("database already contains queued requests")

    attempts = settings.max_queued_requests * 3
    try:
        with TestClient(app) as test_client, ThreadPoolExecutor(max_workers=attempts) as pool:
            codes = list(
                pool.map(
                    lambda _: test_client.post(
                        "/generate", json={"prompt": marker, "max_new_tokens": 1}
                    ).status_code,
                    range(attempts),
                )
            )

        assert codes.count(202) == settings.max_queued_requests
        assert codes.count(503) == attempts - settings.max_queued_requests
        with SessionLocal() as db:
            stored = db.scalar(select(func.count()).where(InferenceRequest.prompt == marker))
        assert stored == settings.max_queued_requests
    finally:
        with SessionLocal() as db:
            db.execute(delete(InferenceRequest).where(InferenceRequest.prompt == marker))
            db.commit()
