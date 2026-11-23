"""Tests for the TEST_REDIS_URL guard (tests/redis_guard.py). No Redis needed."""

from pathlib import Path

import pytest

from app.core.config import Settings
from redis_guard import (
    DEFAULT_DEVELOPMENT_REDIS_URL,
    TEST_REDIS_IDENTITY,
    UnsafeTestRedisError,
    check_redis_identity,
    configure_test_redis,
    resolve_development_redis_url,
    resolve_test_redis_url,
    validate_test_redis_url,
)

TEST_URL = "redis://localhost:16380/0"
DEV_URL = "redis://localhost:16379/0"


@pytest.fixture
def env_file(tmp_path: Path):
    def write(**values: str) -> Path:
        path = tmp_path / ".env"
        path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
        return path

    return write


def test_shell_takes_precedence_then_env_file(env_file):
    path = env_file(TEST_REDIS_URL="redis://localhost:16380/5")

    assert resolve_test_redis_url({"TEST_REDIS_URL": TEST_URL}, path) == TEST_URL
    assert resolve_test_redis_url({}, path) == "redis://localhost:16380/5"


def test_missing_test_redis_url_fails_and_never_uses_redis_url(env_file):
    path = env_file(REDIS_URL=DEV_URL)

    with pytest.raises(UnsafeTestRedisError, match="never fall back to REDIS_URL"):
        resolve_test_redis_url({"REDIS_URL": DEV_URL}, path)


def test_development_url_defaults_to_the_app_default(tmp_path):
    assert resolve_development_redis_url({}, tmp_path / "missing.env") == DEFAULT_DEVELOPMENT_REDIS_URL
    assert DEFAULT_DEVELOPMENT_REDIS_URL == Settings.model_fields["redis_url"].default


def test_valid_test_url_is_accepted():
    assert validate_test_redis_url(TEST_URL, DEV_URL) == TEST_URL


@pytest.mark.parametrize(
    "test_url, dev_url",
    [
        ("redis://localhost:16379/15", DEV_URL),  # another database number is not enough
        ("redis://127.0.0.1:16379/0", DEV_URL),  # loopback aliases
        ("redis://[::1]:16379/0", DEV_URL),
        ("redis://localhost/1", "redis://localhost/0"),  # omitted port means 6379
        ("redis://localhost:6379/1", "redis://localhost/0"),
        ("rediss://Cache.internal:7000/1", "redis://cache.internal:7000/0"),
    ],
)
def test_same_server_as_development_is_rejected(test_url, dev_url):
    with pytest.raises(UnsafeTestRedisError, match="same Redis server as REDIS_URL"):
        validate_test_redis_url(test_url, dev_url)


@pytest.mark.parametrize("bad_url", ["not a url", "http://localhost:16380", "redis://", "redis://localhost:notaport/0"])
def test_malformed_or_non_redis_urls_are_rejected(bad_url):
    with pytest.raises(UnsafeTestRedisError):
        validate_test_redis_url(bad_url, DEV_URL)


def test_configure_points_redis_url_at_test_redis(env_file):
    environ = {"TEST_REDIS_URL": TEST_URL, "REDIS_URL": DEV_URL}

    assert configure_test_redis(environ, env_file()) == TEST_URL
    assert environ["REDIS_URL"] == TEST_URL


def test_configure_rejects_unsafe_url_without_touching_redis_url(env_file):
    environ = {"TEST_REDIS_URL": "redis://localhost:16379/9", "REDIS_URL": DEV_URL}

    with pytest.raises(UnsafeTestRedisError):
        configure_test_redis(environ, env_file())
    assert environ["REDIS_URL"] == DEV_URL


@pytest.mark.parametrize("marker", [TEST_REDIS_IDENTITY, TEST_REDIS_IDENTITY.encode()])
def test_correct_identity_marker_is_accepted(marker):
    check_redis_identity(marker)


@pytest.mark.parametrize("marker", [None, "", b"", "inferenceflow-test-db", "inferenceflow-test-redis ", b"other"])
def test_missing_or_wrong_identity_marker_is_refused(marker):
    with pytest.raises(UnsafeTestRedisError, match="Nothing was deleted"):
        check_redis_identity(marker)


def test_start_script_provisions_exactly_the_expected_marker():
    script = (Path(__file__).resolve().parents[2] / "docker/test-redis/start.sh").read_text()

    assert f"redis-cli SET inferenceflow:test-redis:identity {TEST_REDIS_IDENTITY}" in script
