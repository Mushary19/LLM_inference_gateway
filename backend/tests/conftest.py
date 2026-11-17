import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple

import pytest

from database_guard import (
    IDENTITY_QUERY,
    UnsafeTestDatabaseError,
    check_database_identity,
    configure_test_database,
)
from redis_guard import (
    TEST_REDIS_IDENTITY_KEY,
    UnsafeTestRedisError,
    check_redis_identity,
    configure_test_redis,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Must happen before anything imports the application: it points the app's DATABASE_URL at
# TEST_DATABASE_URL, so every engine, session and migration in this run uses the test database.
if "app.core.config" in sys.modules:
    pytest.exit("app.core.config was imported before the test database guard ran", returncode=4)
try:
    TEST_DATABASE = configure_test_database(os.environ, BACKEND_DIR / ".env")
except UnsafeTestDatabaseError as exc:
    pytest.exit(f"Refusing to run tests: {exc}", returncode=4)

# Same for Redis: point the app's REDIS_URL at TEST_REDIS_URL.
try:
    TEST_REDIS_URL = configure_test_redis(os.environ, BACKEND_DIR / ".env")
except UnsafeTestRedisError as exc:
    pytest.exit(f"Refusing to run tests: {exc}", returncode=4)

import redis  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, delete, select, text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.database.base import Base  # noqa: E402
from app.database.connection import engine, get_db  # noqa: E402
from app.database.models import ApiKey, ClientApplication, QuotaReservation, UsageRecord  # noqa: E402
from app.main import app  # noqa: E402
from app.services.api_keys import create_api_key, create_application  # noqa: E402

if (engine.url.database, engine.url.port) != (TEST_DATABASE.database, TEST_DATABASE.port):
    pytest.exit("the application engine is not bound to TEST_DATABASE_URL", returncode=4)
if settings.redis_url != TEST_REDIS_URL:
    pytest.exit("the application is not configured with TEST_REDIS_URL", returncode=4)

RATE_LIMIT_KEY_PATTERN = "inferenceflow:ratelimit:*"


def delete_rate_limit_keys(client: redis.Redis) -> int:
    """Delete only rate-limit buckets (tests create them), never the identity marker or anything else."""
    deleted = 0
    for key in client.scan_iter(match=RATE_LIMIT_KEY_PATTERN, count=500):
        deleted += client.unlink(key)
    return deleted


@pytest.fixture(scope="session", autouse=True)
def prepared_test_redis() -> Generator[None, None, None]:
    """Once per session: check test-redis is reachable and carries its identity marker, then
    remove leftover rate-limit buckets. The identity check runs before anything is deleted."""
    client = redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=3, socket_timeout=3)
    try:
        try:
            marker = client.get(TEST_REDIS_IDENTITY_KEY)
        except redis.RedisError as exc:
            pytest.exit(
                f"Test Redis is not reachable at {TEST_REDIS_URL}: {exc}\n"
                "Start it with: docker compose up -d --wait test-redis",
                returncode=4,
            )
        try:
            check_redis_identity(marker)
        except UnsafeTestRedisError as exc:
            pytest.exit(f"Refusing to run tests against {TEST_REDIS_URL}: {exc}", returncode=4)
        delete_rate_limit_keys(client)
        yield
        delete_rate_limit_keys(client)
    finally:
        client.close()


@pytest.fixture(scope="session", autouse=True)
def prepared_test_database() -> None:
    """Once per session: check the test database is reachable and is the provisioned test-db,
    then migrate it, then empty it. The identity check runs before anything is modified."""
    where = TEST_DATABASE.render_as_string(hide_password=True)
    probe = create_engine(TEST_DATABASE, connect_args={"connect_timeout": 3})
    try:
        with probe.connect() as connection:
            marker = connection.execute(text(IDENTITY_QUERY)).scalar_one()
    except OperationalError as exc:
        pytest.exit(
            f"Test database is not reachable at {where}: {exc.orig}\n"
            "Start it with: docker compose up -d --wait test-db",
            returncode=4,
        )
    finally:
        probe.dispose()
    try:
        check_database_identity(marker)
    except UnsafeTestDatabaseError as exc:
        pytest.exit(f"Refusing to run tests against {where}: {exc}", returncode=4)

    # No alembic.ini: loading it would reconfigure logging. env.py takes the URL from
    # settings.database_url, which the guard above set to the test database.
    alembic_config = Config()
    alembic_config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    command.upgrade(alembic_config, "head")

    # Application tables only; alembic_version is not part of Base.metadata.
    preparer = engine.dialect.identifier_preparer
    tables = ", ".join(preparer.quote(table.name) for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


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


def delete_committed_application(db: Session, name: str) -> None:
    """Remove an application committed by a test, with everything that references it.

    Its token quota goes with it (ON DELETE CASCADE); usage records, reservations and keys
    don't cascade (they're the audit trail), so they're deleted first, in dependency order.
    """
    application_id = db.scalar(select(ClientApplication.id).where(ClientApplication.name == name))
    if application_id is None:
        return
    db.execute(delete(UsageRecord).where(UsageRecord.application_id == application_id))
    db.execute(delete(QuotaReservation).where(QuotaReservation.application_id == application_id))
    db.execute(delete(ApiKey).where(ApiKey.application_id == application_id))
    db.execute(delete(ClientApplication).where(ClientApplication.id == application_id))
    db.commit()


class IssuedKey(NamedTuple):
    plaintext: str
    record: ApiKey


@pytest.fixture
def api_key(db_session: Session) -> IssuedKey:
    """An active API key for application "test-app", inside the rolled-back test session."""
    application = create_application(db_session, "test-app")
    record, plaintext = create_api_key(db_session, application, "test-key")
    db_session.commit()
    return IssuedKey(plaintext, record)


@pytest.fixture
def auth_headers(api_key: IssuedKey) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key.plaintext}"}


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
