"""Resolve and validate TEST_REDIS_URL before the application is imported by pytest.

Tests only ever talk to the dedicated test-redis server. Like database_guard.py, these
checks run first in conftest.py and any failure stops the whole test run.
"""

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from urllib.parse import urlsplit

from database_guard import _lookup

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_DEFAULT_REDIS_PORT = 6379
# Must match the default of Settings.redis_url (a test checks this): the development Redis
# the app uses when REDIS_URL isn't set.
DEFAULT_DEVELOPMENT_REDIS_URL = "redis://localhost:16379/0"

# Key and value set only by the test-redis container's start script
# (docker/test-redis/start.sh). Pytest verifies it, never sets it.
TEST_REDIS_IDENTITY_KEY = "inferenceflow:test-redis:identity"
TEST_REDIS_IDENTITY = "inferenceflow-test-redis"


class UnsafeTestRedisError(Exception):
    """TEST_REDIS_URL is missing or could point at a non-test Redis server."""


def resolve_test_redis_url(environ: Mapping[str, str], env_file: Path) -> str:
    url = _lookup("TEST_REDIS_URL", environ, env_file)
    if url is None:
        raise UnsafeTestRedisError(
            f"TEST_REDIS_URL is not set (checked the environment and {env_file}). "
            "Tests never fall back to REDIS_URL. Set it to the test-redis container, e.g. "
            "redis://localhost:16380/0"
        )
    return url


def resolve_development_redis_url(environ: Mapping[str, str], env_file: Path) -> str:
    return _lookup("REDIS_URL", environ, env_file) or DEFAULT_DEVELOPMENT_REDIS_URL


def _server(url: str) -> tuple[str, int]:
    """(host, port) of a Redis URL. The logical database number is deliberately ignored."""
    parts = urlsplit(url)
    host = (parts.hostname or "localhost").lower()
    if host in _LOOPBACK_HOSTS:
        host = "localhost"
    return host, parts.port or _DEFAULT_REDIS_PORT


def validate_test_redis_url(test_url: str, development_url: str) -> str:
    try:
        parts = urlsplit(test_url)
        parts.port  # raises ValueError for a malformed port
    except ValueError as exc:
        raise UnsafeTestRedisError(f"TEST_REDIS_URL is not a valid URL: {exc}") from exc
    if parts.scheme not in ("redis", "rediss") or not parts.hostname:
        raise UnsafeTestRedisError("TEST_REDIS_URL must be a redis:// or rediss:// URL with a host")

    try:
        same_server = _server(development_url) == _server(test_url)
    except ValueError:
        same_server = False
    if same_server:
        host, port = _server(test_url)
        raise UnsafeTestRedisError(
            f"TEST_REDIS_URL uses the same Redis server as REDIS_URL ({host}:{port}). A different "
            "database number is not enough; tests must use the separate test-redis container."
        )
    return test_url


def check_redis_identity(marker: bytes | str | None) -> None:
    if isinstance(marker, bytes):
        marker = marker.decode("utf-8", errors="replace")
    if marker != TEST_REDIS_IDENTITY:
        found = "no identity marker" if marker is None else f"identity marker {marker!r}"
        raise UnsafeTestRedisError(
            f"the Redis server at TEST_REDIS_URL has {found}; expected {TEST_REDIS_IDENTITY!r} at "
            f"{TEST_REDIS_IDENTITY_KEY!r}. It is not the InferenceFlow test-redis container. "
            "Nothing was deleted."
        )


def configure_test_redis(environ: MutableMapping[str, str], env_file: Path) -> str:
    """Validate TEST_REDIS_URL and make the application use it via REDIS_URL.

    Must run before anything imports app.core.config (REDIS_URL is read at import time).
    """
    test_url = validate_test_redis_url(
        resolve_test_redis_url(environ, env_file), resolve_development_redis_url(environ, env_file)
    )
    environ["REDIS_URL"] = test_url
    return test_url
