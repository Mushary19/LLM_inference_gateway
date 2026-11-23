"""Resolve and validate TEST_DATABASE_URL before the application is imported by pytest.

Tests must only ever touch the dedicated test database. These checks run first in
conftest.py; any failure stops the whole test run.
"""

from collections.abc import Mapping, MutableMapping
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_DEFAULT_POSTGRES_PORT = 5432

# Database comment set only by the test-db container's init script
# (docker/test-db/initdb/01-identity.sql). Pytest verifies it, never sets it.
TEST_DATABASE_IDENTITY = "inferenceflow-test-db"
IDENTITY_QUERY = (
    "SELECT shobj_description(oid, 'pg_database') FROM pg_database WHERE datname = current_database()"
)


class UnsafeTestDatabaseError(Exception):
    """TEST_DATABASE_URL is missing or could point at a non-test database."""


def _lookup(name: str, environ: Mapping[str, str], env_file: Path) -> str | None:
    """Shell environment first, then the .env file. Empty values count as unset."""
    value = environ.get(name)
    if not value and env_file.is_file():
        value = dotenv_values(env_file).get(name)
    return value or None


def resolve_test_database_url(environ: Mapping[str, str], env_file: Path) -> str:
    url = _lookup("TEST_DATABASE_URL", environ, env_file)
    if url is None:
        raise UnsafeTestDatabaseError(
            f"TEST_DATABASE_URL is not set (checked the environment and {env_file}). "
            "Tests never fall back to DATABASE_URL. Set it to the test-db container, e.g. "
            "postgresql+psycopg://inferenceflow_test:<password>@localhost:15432/inferenceflow_test"
        )
    return url


def resolve_development_database_url(environ: Mapping[str, str], env_file: Path) -> str | None:
    return _lookup("DATABASE_URL", environ, env_file)


def _server(url: URL) -> tuple[str, int]:
    host = (url.host or "localhost").lower()
    if host in _LOOPBACK_HOSTS:
        host = "localhost"
    return host, url.port or _DEFAULT_POSTGRES_PORT


def validate_test_database_url(test_url: str, development_url: str | None) -> URL:
    try:
        url = make_url(test_url)
    except ArgumentError as exc:
        raise UnsafeTestDatabaseError(f"TEST_DATABASE_URL is not a valid database URL: {exc}") from exc

    if not url.drivername.startswith("postgresql"):
        raise UnsafeTestDatabaseError(f"TEST_DATABASE_URL must be a PostgreSQL URL, got {url.drivername!r}")
    if not url.database or not url.database.endswith("_test"):
        raise UnsafeTestDatabaseError(
            f"TEST_DATABASE_URL database name must end with '_test', got {url.database!r}"
        )

    if development_url:
        try:
            dev = make_url(development_url)
        except ArgumentError:
            dev = None
        if dev is not None and _server(dev) == _server(url):
            host, port = _server(url)
            raise UnsafeTestDatabaseError(
                f"TEST_DATABASE_URL uses the same server as DATABASE_URL ({host}:{port}). "
                "Tests must run against the separate test-db container."
            )
    return url


def check_database_identity(marker: str | None) -> None:
    """Refuse any database that isn't the provisioned test-db (wrong server, wrong container)."""
    if marker != TEST_DATABASE_IDENTITY:
        found = "no identity marker" if marker is None else f"identity marker {marker!r}"
        raise UnsafeTestDatabaseError(
            f"the database at TEST_DATABASE_URL has {found}; expected {TEST_DATABASE_IDENTITY!r}. "
            "It is not the InferenceFlow test-db container, or its marker was never provisioned "
            "(see README, 'Testing'). Nothing was migrated or truncated."
        )


def configure_test_database(environ: MutableMapping[str, str], env_file: Path) -> URL:
    """Validate TEST_DATABASE_URL and make the application use it via DATABASE_URL.

    Must run before anything imports app.core.config: the application reads DATABASE_URL
    once, at import time, and environment variables take precedence over .env.
    """
    test_url = resolve_test_database_url(environ, env_file)
    url = validate_test_database_url(test_url, resolve_development_database_url(environ, env_file))
    environ["DATABASE_URL"] = test_url
    return url
