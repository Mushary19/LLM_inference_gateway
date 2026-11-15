"""Tests for the TEST_DATABASE_URL guard (tests/database_guard.py). No database needed."""

from pathlib import Path

import pytest

from app.database.base import Base
from app.database.connection import engine
from database_guard import (
    IDENTITY_QUERY,
    TEST_DATABASE_IDENTITY,
    UnsafeTestDatabaseError,
    check_database_identity,
    configure_test_database,
    resolve_development_database_url,
    resolve_test_database_url,
    validate_test_database_url,
)

TEST_URL = "postgresql+psycopg://inferenceflow_test:pw@localhost:15432/inferenceflow_test"
DEV_URL = "postgresql+psycopg://inferenceflow:pw@localhost:5432/inferenceflow"


@pytest.fixture
def env_file(tmp_path: Path):
    def write(**values: str) -> Path:
        path = tmp_path / ".env"
        path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
        return path

    return write


# --- Resolution: shell first, then .env, never DATABASE_URL ---


def test_shell_environment_takes_precedence_over_env_file(env_file):
    path = env_file(TEST_DATABASE_URL="postgresql+psycopg://u:p@localhost:15432/from_file_test")

    assert resolve_test_database_url({"TEST_DATABASE_URL": TEST_URL}, path) == TEST_URL


def test_env_file_is_used_when_shell_variable_is_absent(env_file):
    path = env_file(TEST_DATABASE_URL=TEST_URL)

    assert resolve_test_database_url({}, path) == TEST_URL


def test_empty_shell_value_falls_through_to_env_file(env_file):
    path = env_file(TEST_DATABASE_URL=TEST_URL)

    assert resolve_test_database_url({"TEST_DATABASE_URL": ""}, path) == TEST_URL


def test_missing_everywhere_fails_clearly(tmp_path):
    with pytest.raises(UnsafeTestDatabaseError, match="TEST_DATABASE_URL is not set"):
        resolve_test_database_url({}, tmp_path / "does-not-exist.env")


def test_never_falls_back_to_database_url(env_file):
    path = env_file(DATABASE_URL=DEV_URL)

    with pytest.raises(UnsafeTestDatabaseError, match="never fall back to DATABASE_URL"):
        resolve_test_database_url({"DATABASE_URL": DEV_URL}, path)


def test_development_url_is_resolved_shell_first(env_file):
    path = env_file(DATABASE_URL="postgresql+psycopg://u:p@localhost:5432/from_file")

    assert resolve_development_database_url({"DATABASE_URL": DEV_URL}, path) == DEV_URL
    assert resolve_development_database_url({}, path).endswith("/from_file")


# --- Validation ---


def test_valid_test_url_is_accepted():
    url = validate_test_database_url(TEST_URL, DEV_URL)

    assert (url.host, url.port, url.database) == ("localhost", 15432, "inferenceflow_test")


@pytest.mark.parametrize(
    "database", ["inferenceflow", "inferenceflow_testing", "test_inferenceflow", "_tests", ""]
)
def test_database_name_must_end_with_test(database):
    with pytest.raises(UnsafeTestDatabaseError, match="must end with '_test'"):
        validate_test_database_url(f"postgresql+psycopg://u:p@localhost:15432/{database}", DEV_URL)


@pytest.mark.parametrize(
    "test_url, dev_url",
    [
        # Same server as development, even with a _test database name.
        ("postgresql+psycopg://u:p@localhost:5432/inferenceflow_test", DEV_URL),
        # Loopback aliases are the same server.
        ("postgresql+psycopg://u:p@127.0.0.1:5432/inferenceflow_test", DEV_URL),
        ("postgresql+psycopg://u:p@[::1]:5432/inferenceflow_test", DEV_URL),
        # An omitted port means 5432.
        ("postgresql+psycopg://u:p@localhost/inferenceflow_test", DEV_URL),
        ("postgresql+psycopg://u:p@localhost:5432/inferenceflow_test", "postgresql+psycopg://u:p@localhost/x"),
        # Same remote host and port.
        ("postgresql+psycopg://u:p@db.internal:6000/app_test", "postgresql+psycopg://u:p@DB.internal:6000/app"),
    ],
)
def test_test_url_on_the_development_server_is_rejected(test_url, dev_url):
    with pytest.raises(UnsafeTestDatabaseError, match="same server as DATABASE_URL"):
        validate_test_database_url(test_url, dev_url)


def test_different_port_on_same_host_is_allowed():
    assert validate_test_database_url(TEST_URL, DEV_URL).port == 15432


def test_validation_without_a_development_url_still_checks_the_name():
    validate_test_database_url(TEST_URL, None)
    with pytest.raises(UnsafeTestDatabaseError):
        validate_test_database_url("postgresql+psycopg://u:p@localhost:15432/inferenceflow", None)


@pytest.mark.parametrize("bad_url", ["not a url", "sqlite:///inferenceflow_test", "mysql://u:p@h/x_test"])
def test_malformed_or_non_postgres_urls_are_rejected(bad_url):
    with pytest.raises(UnsafeTestDatabaseError):
        validate_test_database_url(bad_url, DEV_URL)


def test_configure_points_database_url_at_the_test_database(env_file):
    environ = {"TEST_DATABASE_URL": TEST_URL, "DATABASE_URL": DEV_URL}

    url = configure_test_database(environ, env_file())

    assert environ["DATABASE_URL"] == TEST_URL
    assert url.database == "inferenceflow_test"


def test_configure_rejects_unsafe_url_without_touching_database_url(env_file):
    environ = {"TEST_DATABASE_URL": "postgresql+psycopg://u:p@localhost:5432/inferenceflow_test", "DATABASE_URL": DEV_URL}

    with pytest.raises(UnsafeTestDatabaseError):
        configure_test_database(environ, env_file())
    assert environ["DATABASE_URL"] == DEV_URL


# --- Identity marker ---


def test_correct_identity_marker_is_accepted():
    check_database_identity("inferenceflow-test-db")


@pytest.mark.parametrize(
    "marker, message",
    [
        (None, "no identity marker"),
        ("", "identity marker ''"),
        ("inferenceflow-dev-db", "identity marker 'inferenceflow-dev-db'"),
        ("inferenceflow-test-db ", "identity marker 'inferenceflow-test-db '"),  # exact match only
        ("INFERENCEFLOW-TEST-DB", "identity marker 'INFERENCEFLOW-TEST-DB'"),
    ],
)
def test_missing_or_wrong_identity_marker_is_refused(marker, message):
    with pytest.raises(UnsafeTestDatabaseError, match=message) as excinfo:
        check_database_identity(marker)

    assert "Nothing was migrated or truncated" in str(excinfo.value)


def test_init_script_provisions_exactly_the_expected_marker():
    script = Path(__file__).resolve().parents[2] / "docker/test-db/initdb/01-identity.sql"
    statements = [line for line in script.read_text().splitlines() if line and not line.startswith("--")]

    assert statements == [f"COMMENT ON DATABASE inferenceflow_test IS '{TEST_DATABASE_IDENTITY}';"]


# --- This session ---


def test_this_session_database_carries_the_identity_marker():
    from sqlalchemy import text

    with engine.connect() as connection:
        assert connection.execute(text(IDENTITY_QUERY)).scalar_one() == TEST_DATABASE_IDENTITY


def test_this_session_uses_the_test_database():
    assert engine.url.database.endswith("_test")
    assert engine.url.port != 5432


def test_alembic_version_is_not_an_application_table():
    assert "alembic_version" not in {table.name for table in Base.metadata.sorted_tables}
