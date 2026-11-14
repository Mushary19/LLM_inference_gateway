import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.database.models import InferenceRequest, RequestPriority, RequestStatus


def _stored(db_session: Session, request_id: str) -> InferenceRequest:
    db_session.expire_all()  # force a fresh SELECT instead of the identity map
    row = db_session.get(InferenceRequest, uuid.UUID(request_id))
    assert row is not None
    return row


def test_valid_request_is_stored_and_queued(client, db_session):
    response = client.post(
        "/generate", json={"prompt": "Hello world", "max_new_tokens": 50, "priority": 2}
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "QUEUED"
    assert set(body) == {"request_id", "status"}

    row = _stored(db_session, body["request_id"])
    assert row.prompt == "Hello world"
    assert row.input_tokens == 2
    assert row.max_new_tokens == 50
    assert row.priority == RequestPriority.NORMAL
    assert row.status == RequestStatus.QUEUED
    assert row.created_at is not None
    assert row.started_at is None and row.completed_at is None


def test_prompt_is_stripped_before_storing(client, db_session):
    response = client.post("/generate", json={"prompt": "  Hello world \n", "max_new_tokens": 5})

    assert response.status_code == 202
    row = _stored(db_session, response.json()["request_id"])
    assert row.prompt == "Hello world"
    assert row.input_tokens == 2


def test_priority_defaults_to_normal(client, db_session):
    response = client.post("/generate", json={"prompt": "Hello world", "max_new_tokens": 50})

    assert response.status_code == 202
    assert _stored(db_session, response.json()["request_id"]).priority == RequestPriority.NORMAL


@pytest.mark.parametrize("priority", [1, 3])
def test_other_valid_priorities(client, db_session, priority):
    response = client.post(
        "/generate", json={"prompt": "Hello", "max_new_tokens": 1, "priority": priority}
    )

    assert response.status_code == 202
    assert _stored(db_session, response.json()["request_id"]).priority == priority


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t "])
def test_blank_prompt_is_rejected(client, prompt):
    response = client.post("/generate", json={"prompt": prompt, "max_new_tokens": 50})

    assert response.status_code == 422


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_non_positive_max_new_tokens_is_rejected(client, max_new_tokens):
    response = client.post("/generate", json={"prompt": "Hello", "max_new_tokens": max_new_tokens})

    assert response.status_code == 422


@pytest.mark.parametrize("priority", [0, 4, -1])
def test_out_of_range_priority_is_rejected(client, priority):
    response = client.post(
        "/generate", json={"prompt": "Hello", "max_new_tokens": 50, "priority": priority}
    )

    assert response.status_code == 422


def test_rejected_requests_are_not_stored(client, db_session):
    client.post("/generate", json={"prompt": "   ", "max_new_tokens": 50})
    client.post("/generate", json={"prompt": "Hello", "max_new_tokens": 0})

    assert db_session.scalar(select(func.count()).select_from(InferenceRequest)) == 0


def test_database_failure_rolls_back_and_returns_500(client, db_session, monkeypatch):
    def failing_commit():
        db_session.flush()  # the INSERT reaches PostgreSQL, then the commit fails
        raise OperationalError("COMMIT", {}, Exception("simulated database failure"))

    monkeypatch.setattr(db_session, "commit", failing_commit)

    response = client.post("/generate", json={"prompt": "Hello world", "max_new_tokens": 50})

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to store inference request"}
    # The flushed INSERT was rolled back and the session is still usable.
    assert db_session.scalar(select(func.count()).select_from(InferenceRequest)) == 0


def test_max_new_tokens_above_generation_limit_is_rejected(client, db_session):
    from app.core.config import settings

    at_limit = client.post(
        "/generate", json={"prompt": "Hello", "max_new_tokens": settings.max_generation_tokens}
    )
    above = client.post(
        "/generate", json={"prompt": "Hello", "max_new_tokens": settings.max_generation_tokens + 1}
    )

    assert at_limit.status_code == 202
    assert above.status_code == 422
