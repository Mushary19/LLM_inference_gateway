"""HTTP schemas for GET /v1/usage and GET /v1/usage/quota."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class UsageFiltersOut(BaseModel):
    start: datetime | None
    end: datetime | None
    model: str | None
    provider: str | None


class UsageSummaryOut(BaseModel):
    requests: int
    # Requests the provider reported no usage for; charged their full reservation.
    requests_without_reported_usage: int
    # Provider-reported token counts only.
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # What was deducted from the quota (includes conservative charges for unreported usage).
    charged_tokens: int


class UsageRecordOut(BaseModel):
    request_id: str  # the chat completion ID returned to the client
    created_at: datetime
    key_prefix: str
    provider: str
    model: str
    status: Literal["succeeded", "failed", "expired"]
    usage_reported: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    charged_tokens: int
    latency_ms: float | None


class PaginationOut(BaseModel):
    limit: int
    offset: int
    has_more: bool


class UsageReport(BaseModel):
    object: Literal["usage"] = "usage"
    filters: UsageFiltersOut
    summary: UsageSummaryOut
    data: list[UsageRecordOut]
    pagination: PaginationOut


class QuotaReport(BaseModel):
    object: Literal["quota"] = "quota"
    limit_tokens: int
    consumed_tokens: int
    reserved_tokens: int
    available_tokens: int
