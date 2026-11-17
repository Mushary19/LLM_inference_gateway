"""Usage endpoints: GET /v1/usage and GET /v1/usage/quota.

On the shared /v1 router, so they are authenticated and rate-limited like every /v1 route.
Results are always scoped to the authenticated key's application. Plain `def` routes, so
FastAPI runs the synchronous queries in a worker thread; each ends its read transaction
before returning.
"""

from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.auth import AuthenticatedClient, require_api_key
from app.api.v1 import router
from app.database.connection import get_db
from app.database.models import TokenQuota
from app.schemas.usage import PaginationOut, QuotaReport, UsageFiltersOut, UsageRecordOut, UsageReport, UsageSummaryOut
from app.services.usage import UsageFilters, usage_records, usage_summary


def _as_utc(value: datetime | None) -> datetime | None:
    """Times without an offset are taken as UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@router.get("/usage", response_model=UsageReport)
def get_usage(
    start: datetime | None = Query(None, description="Inclusive, ISO 8601; UTC if no offset"),
    end: datetime | None = Query(None, description="Exclusive, ISO 8601; UTC if no offset"),
    model: str | None = Query(None),
    provider: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    client: AuthenticatedClient = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> UsageReport:
    filters = UsageFilters(start=_as_utc(start), end=_as_utc(end), model=model, provider=provider)
    if filters.start and filters.end and filters.start >= filters.end:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="start must be before end")

    summary = usage_summary(db, client.application_id, filters)
    page, has_more = usage_records(db, client.application_id, filters, limit, offset)
    report = UsageReport(
        filters=UsageFiltersOut(start=filters.start, end=filters.end, model=model, provider=provider),
        summary=UsageSummaryOut(**summary.__dict__),
        data=[
            UsageRecordOut(
                request_id=f"chatcmpl-{record.request_id.hex}",
                created_at=record.created_at,
                key_prefix=key_prefix,
                provider=record.provider,
                model=record.model,
                status=record.status.value,
                usage_reported=record.usage_reported,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                total_tokens=record.total_tokens,
                charged_tokens=record.charged_tokens,
                latency_ms=record.latency_ms,
            )
            for record, key_prefix in page
        ],
        pagination=PaginationOut(limit=limit, offset=offset, has_more=has_more),
    )
    db.rollback()  # end the read transaction
    return report


@router.get("/usage/quota", response_model=QuotaReport)
def get_quota(
    client: AuthenticatedClient = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> QuotaReport:
    quota = db.get(TokenQuota, client.application_id)
    if quota is None:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No token quota is configured for this application")
    report = QuotaReport(
        limit_tokens=quota.limit_tokens,
        consumed_tokens=quota.consumed_tokens,
        reserved_tokens=quota.reserved_tokens,
        available_tokens=quota.available_tokens,
    )
    db.rollback()  # end the read transaction
    return report
