"""API key authentication for the gateway endpoints (Authorization: Bearer <api_key>)."""

import logging
import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.services.api_keys import authenticate

logger = logging.getLogger(__name__)

# auto_error=False: missing or non-Bearer credentials come through as None, so every failure
# gets the same 401 below. Also registers the Bearer scheme in the OpenAPI docs.
bearer_scheme = HTTPBearer(auto_error=False, description="InferenceFlow API key (if_...)")


@dataclass(frozen=True)
class AuthenticatedClient:
    api_key_id: uuid.UUID
    key_prefix: str
    key_name: str
    application_id: uuid.UUID
    application_name: str


def _unauthorized() -> HTTPException:
    # One message for missing, malformed, unknown and revoked keys alike.
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> AuthenticatedClient:
    """Resolve the Bearer API key to its client application, or reject with 401.

    A plain `def` dependency, so FastAPI runs the synchronous database lookup in a worker
    thread. The presented key is never logged.
    """
    if credentials is None:
        raise _unauthorized()
    api_key = authenticate(db, credentials.credentials)
    if api_key is None:
        logger.info("Rejected request with an invalid or revoked API key")
        raise _unauthorized()
    client = AuthenticatedClient(
        api_key_id=api_key.id,
        key_prefix=api_key.key_prefix,
        key_name=api_key.name,
        application_id=api_key.application_id,
        application_name=api_key.application.name,
    )
    # End the read transaction now: the request-scoped session otherwise keeps its pooled
    # connection "idle in transaction" for the whole (possibly long) provider call. The
    # client above holds plain values only, so nothing reads the session after this.
    db.rollback()
    return client
