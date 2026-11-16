"""Rate-limit dependency for the /v1 API. Runs after authentication, before the route."""

import logging
import math

from fastapi import Depends, HTTPException, Request, Response, status

from app.api.auth import AuthenticatedClient, require_api_key
from app.services.rate_limit import RateLimiterUnavailableError, RateLimitResult, TokenBucketRateLimiter

logger = logging.getLogger(__name__)


def get_rate_limiter(request: Request) -> TokenBucketRateLimiter:
    """The limiter created at startup. Tests override this dependency."""
    return request.app.state.rate_limiter


def _headers(result: RateLimitResult) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        # Seconds until the bucket is full again.
        "X-RateLimit-Reset": str(math.ceil(result.reset_after_seconds)),
    }


async def enforce_rate_limit(
    response: Response,
    client: AuthenticatedClient = Depends(require_api_key),
    limiter: TokenBucketRateLimiter = Depends(get_rate_limiter),
) -> None:
    """Take one token from the authenticated key's bucket, or reject with 429.

    Depends on require_api_key, so it only runs for authenticated requests; FastAPI's
    per-request dependency cache means the key is still looked up only once. Fails closed:
    if Redis can't be reached the request is refused with 503, never let through.
    """
    try:
        result = await limiter.consume(str(client.api_key_id))
    except RateLimiterUnavailableError as exc:
        logger.error("Rate limiting unavailable for key %s: %s", client.key_prefix, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiting is temporarily unavailable",
        ) from exc

    headers = _headers(result)
    if not result.allowed:
        logger.info("Rate limit exceeded for key %s", client.key_prefix)
        retry_after = max(1, math.ceil(result.retry_after_seconds))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={**headers, "Retry-After": str(retry_after)},
        )
    response.headers.update(headers)
