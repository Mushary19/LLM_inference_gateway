"""GET /health: liveness of the API process only.

Deliberately checks nothing else: no database, Redis or provider. An orchestrator must never
restart a healthy gateway because an upstream provider is down. Provider health and model
availability are checked on demand with `python -m app.cli check-providers`.

Not under /v1, so it needs no API key and isn't rate-limited.
"""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()
