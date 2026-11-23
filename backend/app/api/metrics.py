"""GET /metrics: the application's Prometheus metrics, for scraping.

Disabled unless METRICS_TOKEN is set: it then answers 404, exactly like an unknown path.
When enabled, it requires `Authorization: Bearer <METRICS_TOKEN>` (compared in constant
time) and answers 401 otherwise, without any metrics.

Deliberately separate from the gateway API: not under /v1, so no API key, rate limit or
quota is involved, and not in the OpenAPI schema. It has no dependencies, so a scrape never
touches the database, Redis or a provider: it only serializes in-memory values. The token
is never logged (request headers aren't logged anywhere).
"""

import hmac

from fastapi import APIRouter, HTTPException, Request, Response, status

from app.core.config import settings
from app.core.metrics import CONTENT_TYPE, METRICS_PATH, metrics_of

router = APIRouter()


def _authorized(authorization: str | None, token: str) -> bool:
    if authorization is None:
        return False
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return False
    return hmac.compare_digest(credentials.encode("utf-8"), token.encode("utf-8"))


@router.get(METRICS_PATH, include_in_schema=False)
async def metrics(request: Request) -> Response:
    token = settings.metrics_token
    if token is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)  # same response as an unknown path
    if not _authorized(request.headers.get("authorization"), token.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid metrics token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Response(content=metrics_of(request.app).exposition(), media_type=CONTENT_TYPE)
