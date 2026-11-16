"""TokenBucketRateLimiter against the real test-redis server.

Algorithm tests pass explicit timestamps (now_us), so refill and retry timing are exact and
nothing sleeps. Concurrency tests use the production path (Redis TIME) with genuinely
concurrent calls; a control test shows the same harness catches a non-atomic implementation.
"""

import asyncio
import math
import socket
import threading
import uuid

import pytest
from redis.asyncio import Redis

from app.core.config import settings
from app.services.rate_limit import (
    KEY_NAMESPACE,
    RateLimiterUnavailableError,
    TokenBucketRateLimiter,
    bucket_key,
    create_redis_client,
)

T0 = 1_700_000_000_000_000  # microseconds
SECOND = 1_000_000


def redis_client(url: str | None = None, timeout: float = 2.0) -> Redis:
    """Built exactly as in production (blocking pool, no retries), against test-redis."""
    return create_redis_client(url or settings.redis_url, timeout)  # TEST_REDIS_URL, via the guard


def run(scenario, rpm: int = 60, burst: int = 10, **client_kwargs):
    """Run `scenario(limiter, redis)` on a fresh client in its own event loop."""

    async def main():
        client = redis_client(**client_kwargs)
        try:
            return await scenario(TokenBucketRateLimiter(client, rpm, burst), client)
        finally:
            await client.aclose()

    return asyncio.run(main())


def new_bucket() -> str:
    return str(uuid.uuid4())


# --- Algorithm (deterministic timestamps) ---


def test_first_request_is_accepted_from_a_full_bucket():
    async def scenario(limiter, _):
        return await limiter.consume(new_bucket(), now_us=T0)

    result = run(scenario)

    assert result.allowed and result.limit == 10
    assert result.remaining == 9 and result.tokens == pytest.approx(9)
    assert result.retry_after_seconds == 0
    assert result.reset_after_seconds == pytest.approx(1.0)  # 1 token at 1 token/s


def test_requests_consume_tokens_until_the_bucket_is_empty():
    async def scenario(limiter, _):
        bucket = new_bucket()
        return [await limiter.consume(bucket, now_us=T0) for _ in range(11)]

    results = run(scenario)

    assert [r.allowed for r in results] == [True] * 10 + [False]
    assert [r.remaining for r in results] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0]
    assert results[-1].retry_after_seconds == pytest.approx(1.0)
    assert results[-1].reset_after_seconds == pytest.approx(10.0)


def test_tokens_refill_continuously():
    async def scenario(limiter, _):
        bucket = new_bucket()
        for _ in range(10):
            await limiter.consume(bucket, now_us=T0)
        half = await limiter.consume(bucket, now_us=T0 + SECOND // 2)  # 0.5 tokens: rejected
        one = await limiter.consume(bucket, now_us=T0 + SECOND)  # 1.0 token: accepted
        later = await limiter.consume(bucket, now_us=T0 + SECOND * 7 // 2)  # +2.5 tokens
        return half, one, later

    half, one, later = run(scenario)

    assert not half.allowed and half.retry_after_seconds == pytest.approx(0.5)
    assert one.allowed and one.tokens == pytest.approx(0)
    assert later.allowed and later.tokens == pytest.approx(1.5) and later.remaining == 1


def test_refill_never_exceeds_burst_capacity():
    async def scenario(limiter, _):
        bucket = new_bucket()
        await limiter.consume(bucket, now_us=T0)
        return await limiter.consume(bucket, now_us=T0 + 3600 * SECOND)

    result = run(scenario)

    assert result.allowed and result.tokens == pytest.approx(9)


@pytest.mark.parametrize("rpm, expected_retry", [(60, 1.0), (30, 2.0), (120, 0.5), (1, 60.0)])
def test_retry_after_follows_the_refill_rate(rpm, expected_retry):
    async def scenario(limiter, _):
        bucket = new_bucket()
        await limiter.consume(bucket, now_us=T0)
        return await limiter.consume(bucket, now_us=T0)

    result = run(scenario, rpm=rpm, burst=1)

    assert not result.allowed and result.retry_after_seconds == pytest.approx(expected_retry)


def test_reset_is_time_until_the_bucket_is_full():
    async def scenario(limiter, _):
        bucket = new_bucket()
        return [await limiter.consume(bucket, now_us=T0) for _ in range(3)][-1]

    result = run(scenario, rpm=30, burst=10)  # 0.5 tokens/s, 3 tokens missing

    assert result.reset_after_seconds == pytest.approx(6.0)


def test_fractional_refill_reaches_exactly_one_token():
    async def scenario(limiter, _):
        bucket = new_bucket()
        await limiter.consume(bucket, now_us=T0)
        just_before = await limiter.consume(bucket, now_us=T0 + math.floor(60 / 7 * SECOND) - 1000)
        exactly = await limiter.consume(bucket, now_us=T0 + round(60 / 7 * SECOND))
        return just_before, exactly

    just_before, exactly = run(scenario, rpm=7, burst=1)

    assert not just_before.allowed
    assert exactly.allowed  # floating-point error must not reject a fully refilled token


def test_clock_moving_backwards_does_not_refill_or_corrupt_the_bucket():
    async def scenario(limiter, client):
        bucket = new_bucket()
        await limiter.consume(bucket, now_us=T0)
        result = await limiter.consume(bucket, now_us=T0 - 10 * SECOND)
        stored = await client.hgetall(bucket_key(bucket))
        return result, stored

    result, stored = run(scenario, burst=2)

    assert result.allowed and result.tokens == pytest.approx(0)
    assert int(float(stored[b"updated_us"])) == T0  # the later timestamp is kept


def test_buckets_are_independent():
    async def scenario(limiter, _):
        a, b = new_bucket(), new_bucket()
        for _ in range(3):
            await limiter.consume(a, now_us=T0)
        return await limiter.consume(a, now_us=T0), await limiter.consume(b, now_us=T0)

    a_result, b_result = run(scenario, burst=3)

    assert not a_result.allowed
    assert b_result.allowed and b_result.remaining == 2


def test_bucket_is_stored_under_the_namespaced_key_with_a_ttl():
    async def scenario(limiter, client):
        bucket = new_bucket()
        await limiter.consume(bucket, now_us=T0)
        key = f"{KEY_NAMESPACE}:{bucket}"
        return limiter.ttl_seconds, key, await client.type(key), await client.hgetall(key), await client.ttl(key)

    ttl_setting, key, key_type, fields, ttl = run(scenario, rpm=60, burst=10)

    assert key.startswith("inferenceflow:ratelimit:")
    assert key_type == b"hash" and set(fields) == {b"tokens", b"updated_us"}
    # A full refill takes 10 s, so the idle bucket expires one second after that.
    assert ttl_setting == 11
    assert 0 < ttl <= 11


def test_production_path_uses_redis_server_time():
    async def scenario(limiter, client):
        bucket = new_bucket()
        before = await client.time()
        result = await limiter.consume(bucket)  # no now_us
        after = await client.time()
        stored = await client.hgetall(bucket_key(bucket))
        return result, before, after, stored

    result, before, after, stored = run(scenario)

    updated = int(float(stored[b"updated_us"]))
    assert result.allowed
    assert before[0] * SECOND + before[1] <= updated <= after[0] * SECOND + after[1]


@pytest.mark.parametrize("rpm, burst", [(0, 10), (60, 0), (-1, 5)])
def test_invalid_limits_are_rejected(rpm, burst):
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(redis_client(), rpm, burst)


@pytest.mark.parametrize(
    "raw",
    [None, [], [1, b"1"], [2, b"1", b"0", b"0"], [1, b"x", b"0", b"0"], [1, b"-1", b"0", b"0"], [1, b"nan", b"0", b"0"], "OK"],
)
def test_unexpected_script_responses_fail_closed(raw):
    limiter = TokenBucketRateLimiter(redis_client(), 60, 10)

    with pytest.raises(RateLimiterUnavailableError, match="unexpected response"):
        limiter._parse(raw)


# --- Redis failures ---


@pytest.fixture
def closed_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]  # nothing listens here once the socket is closed


@pytest.fixture
def stalled_port():
    """A TCP server that accepts connections and never answers, so commands time out."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    held, stop = [], threading.Event()

    def accept():
        server.settimeout(0.1)
        while not stop.is_set():
            try:
                held.append(server.accept()[0])
            except OSError:
                pass

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    thread.join()
    for connection in held:
        connection.close()
    server.close()


def test_unreachable_redis_raises_rate_limiter_unavailable(closed_port):
    async def scenario(limiter, _):
        await limiter.consume(new_bucket())

    with pytest.raises(RateLimiterUnavailableError, match="ConnectionError"):
        run(scenario, url=f"redis://127.0.0.1:{closed_port}/0", timeout=0.5)


def test_redis_timeout_raises_rate_limiter_unavailable(stalled_port):
    async def scenario(limiter, _):
        await limiter.consume(new_bucket())

    with pytest.raises(RateLimiterUnavailableError, match="TimeoutError"):
        run(scenario, url=f"redis://127.0.0.1:{stalled_port}/0", timeout=0.2)


# --- Concurrency against real Redis (production path: Redis TIME) ---


@pytest.mark.parametrize("burst, attempts", [(1, 50), (20, 200), (50, 500)])
def test_concurrent_requests_never_exceed_bucket_capacity(burst, attempts):
    # 1 request/minute: refill during the test (well under a second) is negligible.
    async def scenario(limiter, _):
        bucket = new_bucket()
        results = await asyncio.gather(*(limiter.consume(bucket) for _ in range(attempts)))
        return [r.allowed for r in results]

    allowed = run(scenario, rpm=1, burst=burst)

    assert allowed.count(True) == burst
    assert allowed.count(False) == attempts - burst


def test_concurrent_requests_from_separate_clients_share_one_bucket():
    """Several independent clients (like separate FastAPI workers) hitting one bucket."""

    async def main():
        clients = [redis_client() for _ in range(5)]
        try:
            limiters = [TokenBucketRateLimiter(c, 1, 25) for c in clients]
            bucket = new_bucket()
            results = await asyncio.gather(*(limiters[i % 5].consume(bucket) for i in range(250)))
            return [r.allowed for r in results]
        finally:
            for c in clients:
                await c.aclose()

    allowed = asyncio.run(main())

    assert allowed.count(True) == 25


def test_control_a_non_atomic_get_then_set_limiter_is_caught_by_this_harness():
    """Proves the concurrency tests can detect races: the naive version over-admits."""

    async def naive_consume(client: Redis, key: str, capacity: int) -> bool:
        tokens = await client.get(key)
        tokens = capacity if tokens is None else int(tokens)
        if tokens < 1:
            return False
        await client.set(key, tokens - 1, ex=60)
        return True

    async def main():
        client = redis_client()
        key = f"{KEY_NAMESPACE}:naive-{uuid.uuid4()}"
        try:
            return await asyncio.gather(*(naive_consume(client, key, 5) for _ in range(100)))
        finally:
            await client.delete(key)
            await client.aclose()

    allowed = asyncio.run(main())

    assert allowed.count(True) > 5  # races let far more than 5 requests through
