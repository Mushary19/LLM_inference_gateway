"""Pydantic schemas for inference requests and responses."""

import uuid
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator

from app.core.config import settings
from app.database.models import RequestPriority, RequestStatus


class GenerateRequest(BaseModel):
    # Whitespace is stripped before the length check, so "   " is rejected.
    prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    max_new_tokens: int = Field(gt=0)
    priority: RequestPriority = RequestPriority.NORMAL

    @field_validator("max_new_tokens")
    @classmethod
    def _within_generation_limit(cls, value: int) -> int:
        if value > settings.max_generation_tokens:
            raise ValueError(f"max_new_tokens must be <= {settings.max_generation_tokens}")
        return value


class GenerateResponse(BaseModel):
    request_id: uuid.UUID
    status: RequestStatus
