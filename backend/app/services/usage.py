"""Read-only usage queries, always scoped to one client application."""

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.database.models import ApiKey, UsageRecord


@dataclass(frozen=True)
class UsageFilters:
    start: datetime | None = None  # inclusive
    end: datetime | None = None  # exclusive
    model: str | None = None
    provider: str | None = None


@dataclass(frozen=True)
class UsageSummary:
    requests: int
    requests_without_reported_usage: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    charged_tokens: int


def _filtered(query: Select, application_id: uuid.UUID, filters: UsageFilters) -> Select:
    query = query.where(UsageRecord.application_id == application_id)
    if filters.start is not None:
        query = query.where(UsageRecord.created_at >= filters.start)
    if filters.end is not None:
        query = query.where(UsageRecord.created_at < filters.end)
    if filters.model is not None:
        query = query.where(UsageRecord.model == filters.model)
    if filters.provider is not None:
        query = query.where(UsageRecord.provider == filters.provider)
    return query


def usage_summary(session: Session, application_id: uuid.UUID, filters: UsageFilters) -> UsageSummary:
    """Totals over all matching records. Token sums only include provider-reported counts."""
    row = session.execute(
        _filtered(
            select(
                func.count(),
                func.count().filter(UsageRecord.usage_reported.is_(False)),
                func.coalesce(func.sum(UsageRecord.prompt_tokens), 0),
                func.coalesce(func.sum(UsageRecord.completion_tokens), 0),
                func.coalesce(func.sum(UsageRecord.total_tokens), 0),
                func.coalesce(func.sum(UsageRecord.charged_tokens), 0),
            ),
            application_id,
            filters,
        )
    ).one()
    return UsageSummary(*(int(v) for v in row))


def usage_records(
    session: Session, application_id: uuid.UUID, filters: UsageFilters, limit: int, offset: int
) -> tuple[list[tuple[UsageRecord, str]], bool]:
    """A page of (record, key prefix), newest first, and whether more records follow."""
    rows = session.execute(
        _filtered(
            select(UsageRecord, ApiKey.key_prefix).join(ApiKey, ApiKey.id == UsageRecord.api_key_id),
            application_id,
            filters,
        )
        .order_by(UsageRecord.created_at.desc(), UsageRecord.id.desc())
        .limit(limit + 1)
        .offset(offset)
    ).all()
    return [(record, prefix) for record, prefix in rows[:limit]], len(rows) > limit
