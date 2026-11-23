"""Per-API-key rate limiting with a token bucket stored in Redis.

Each bucket holds up to `burst` tokens and refills continuously at
`requests_per_minute / 60` tokens per second. A request takes one token, or is rejected
when less than one is left.

The whole read-refill-consume-write cycle runs as one Lua script, which Redis executes
atomically: concurrent requests, from any number of workers, can never take more tokens
than the bucket holds. The script reads the current time from Redis itself (TIME), so
workers with different clocks still agree.
"""

import math
from dataclasses import dataclass

from redis.asyncio import BlockingConnectionPool, Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

KEY_NAMESPACE = "inferenceflow:ratelimit"
REDIS_CLIENT_NAME = "inferenceflow-ratelimit"
# Per process. Each check is one sub-millisecond script call, so a small pool serves many
# concurrent requests; extra requests wait (up to the Redis timeout) for a free connection.
REDIS_MAX_CONNECTIONS = 50

# KEYS[1] bucket key
# ARGV[1] capacity (tokens)
# ARGV[2] refill rate (tokens per second)
# ARGV[3] TTL in seconds
# ARGV[4] current time in microseconds, or "" to use Redis TIME (production always uses "")
# Returns {allowed (1/0), tokens left, seconds until a token is available, seconds until full}.
_TOKEN_BUCKET_SCRIPT = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local now
if ARGV[4] ~= "" then
    now = tonumber(ARGV[4])
else
    local time = redis.call("TIME")
    now = tonumber(time[1]) * 1000000 + tonumber(time[2])
end

local state = redis.call("HMGET", KEYS[1], "tokens", "updated_us")
local tokens = tonumber(state[1])
local updated = tonumber(state[2])
if tokens == nil or updated == nil then
    tokens = capacity
    updated = now
end

-- Never refill backwards if the clock moved back; keep the latest timestamp.
local elapsed = math.max(0, now - updated)
tokens = math.min(capacity, tokens + elapsed / 1000000 * rate)
updated = math.max(updated, now)

local allowed = 0
local retry_after = 0
-- The epsilon absorbs floating-point error from fractional refills.
if tokens + 1e-9 >= 1 then
    tokens = math.max(0, tokens - 1)
    allowed = 1
else
    retry_after = (1 - tokens) / rate
end
local reset_after = (capacity - tokens) / rate

redis.call("HSET", KEYS[1], "tokens", string.format("%.17g", tokens), "updated_us", string.format("%.17g", updated))
redis.call("EXPIRE", KEYS[1], ttl)

-- Lua numbers would be truncated to integers on the way out, so return strings.
return {allowed, string.format("%.6f", tokens), string.format("%.6f", retry_after), string.format("%.6f", reset_after)}
"""


def create_redis_client(url: str, timeout_seconds: float, max_connections: int = REDIS_MAX_CONNECTIONS) -> Redis:
    """A Redis client owning one connection pool, to be shared for the life of the process.

    Creating it doesn't connect. The pool blocks (rather than failing) when all connections
    are busy, for at most `timeout_seconds`. No retries: an outage should fail fast as 503,
    not stack several timeouts per request.
    """
    pool = BlockingConnectionPool.from_url(
        url,
        max_connections=max_connections,
        timeout=timeout_seconds,
        socket_connect_timeout=timeout_seconds,
        socket_timeout=timeout_seconds,
        retry=Retry(NoBackoff(), retries=0),
        client_name=REDIS_CLIENT_NAME,
    )
    return Redis.from_pool(pool)  # the client owns the pool and closes it in aclose()


class RateLimiterUnavailableError(Exception):
    """The rate-limit check could not be completed (Redis unreachable, slow or misbehaving)."""


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int  # bucket capacity
    tokens: float  # tokens left after this request
    retry_after_seconds: float  # 0 when allowed
    reset_after_seconds: float  # until the bucket is full again

    @property
    def remaining(self) -> int:
        return math.floor(self.tokens + 1e-9)


def bucket_key(bucket_id: str) -> str:
    return f"{KEY_NAMESPACE}:{bucket_id}"


class TokenBucketRateLimiter:
    def __init__(self, redis: Redis, requests_per_minute: int, burst: int) -> None:
        if requests_per_minute <= 0 or burst <= 0:
            raise ValueError("requests_per_minute and burst must be > 0")
        self.redis = redis
        self.capacity = burst
        self.refill_per_second = requests_per_minute / 60
        # An idle bucket is full again after capacity / rate seconds; expiring it then is
        # equivalent to keeping it, so inactive buckets don't accumulate.
        self.ttl_seconds = math.ceil(self.capacity / self.refill_per_second) + 1
        # Registered locally only; sent with EVALSHA (falling back to EVAL) on first use.
        self._script = redis.register_script(_TOKEN_BUCKET_SCRIPT)

    async def consume(self, bucket_id: str, *, now_us: int | None = None) -> RateLimitResult:
        """Take one token from `bucket_id`'s bucket.

        `now_us` exists only for deterministic tests; production callers never pass it, so
        the script uses Redis server time.
        """
        args = [self.capacity, repr(self.refill_per_second), self.ttl_seconds, "" if now_us is None else now_us]
        try:
            raw = await self._script(keys=[bucket_key(bucket_id)], args=args)
        except RedisError as exc:
            raise RateLimiterUnavailableError(f"rate-limit check failed ({type(exc).__name__})") from exc
        return self._parse(raw)

    def _parse(self, raw: object) -> RateLimitResult:
        try:
            allowed, tokens, retry_after, reset_after = raw  # type: ignore[misc]
            if allowed not in (0, 1):
                raise ValueError(f"unexpected allowed flag {allowed!r}")
            values = [float(v) for v in (tokens, retry_after, reset_after)]
        except (TypeError, ValueError) as exc:
            raise RateLimiterUnavailableError("unexpected response from the rate-limit script") from exc
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise RateLimiterUnavailableError("unexpected response from the rate-limit script")
        return RateLimitResult(
            allowed=allowed == 1,
            limit=self.capacity,
            tokens=values[0],
            retry_after_seconds=values[1],
            reset_after_seconds=values[2],
        )

    async def aclose(self) -> None:
        await self.redis.aclose()
