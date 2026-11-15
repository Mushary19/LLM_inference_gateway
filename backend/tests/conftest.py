from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import engine, get_db
from app.main import app


@pytest.fixture(autouse=True)
def scheduler_disabled(monkeypatch):
    """Keep the background scheduler off unless a test enables it explicitly.

    Otherwise TestClient(app) would start it and it would claim rows other tests commit.
    """
    monkeypatch.setattr(settings, "scheduler_enabled", False)


@pytest.fixture(autouse=True)
def simulated_inference_engine(monkeypatch):
    """Never load a real model in tests, even if .env sets INFERENCE_ENGINE=huggingface."""
    monkeypatch.setattr(settings, "inference_engine", "simulated")


@pytest.fixture(autouse=True)
def fake_chat_provider(monkeypatch):
    """Never call a real provider (or send a real API key) from tests, whatever .env configures."""
    monkeypatch.setattr(settings, "chat_provider", "fake")
    monkeypatch.setattr(settings, "openai_compatible_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "openai_compatible_api_key", None)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Session inside an outer transaction that is rolled back after each test.

    The endpoint's commit() only releases a savepoint, so nothing is persisted.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        join_transaction_mode="create_savepoint",
        autoflush=False,
        expire_on_commit=False,
    )
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
