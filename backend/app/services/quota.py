"""Token quota reservation and settlement (usage accounting).

Lifecycle of one inference request:
1. reserve()  - one transaction: lock the application's quota row (SELECT ... FOR UPDATE),
                check availability, insert a `reserved` reservation, add to reserved_tokens,
                commit. Committing releases the connection before the provider is called.
2. provider call, with no database transaction or connection held.
3. settle()   - one transaction: lock the quota row, then the reservation; if it is still
                `reserved`, move its tokens from reserved_tokens to consumed_tokens (charging
                what the policy below says), write the usage record, commit. If it was already
                settled, do nothing: a request can never be charged twice.

This is a SOFT quota: it is checked against an estimate before the provider runs, and the
actual usage is then charged even if it exceeds the estimate (and so the limit). Accounting
stays accurate; availability is clamped at zero but consumption is never hidden.

Charging policy:
- the provider reported usage            -> charge the actual total (even above the reservation)
- no usage, and the provider may have run -> charge the full reservation, flagged as unreported
- the provider certainly did nothing      -> release: charge nothing, no usage record
- never settled (crash, cancellation, DB failure) -> stays `reserved` until
  expire_stale_reservations() settles it as `expired`, charging the full reservation.

Lock order is always quota row, then reservation row, so concurrent settlements can't deadlock.
"""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.models import QuotaReservation, ReservationStatus, TokenQuota, UsageRecord, UsageStatus
from app.providers.base import ChatMessage, Usage

logger = logging.getLogger(__name__)

# Estimation constants. The prompt is counted as UTF-8 bytes: byte-level BPE tokenizers (GPT,
# Llama 3, Qwen, ...) never produce more tokens than bytes, so this over- rather than
# under-reserves for them. Other tokenizers are not guaranteed to stay under it.
MESSAGE_OVERHEAD_TOKENS = 16  # role markers / chat-template tokens per message
REQUEST_OVERHEAD_TOKENS = 64  # template framing, default system prompts (Qwen adds ~30)


class InsufficientQuotaError(Exception):
    def __init__(self, required_tokens: int, available_tokens: int, reason: str) -> None:
        super().__init__(reason)
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens


class QuotaNotFoundError(Exception):
    """The application's quota row is missing, so a reservation can't be settled.

    Only possible if the row was deleted by hand. The reservation is left untouched (still
    `reserved`); restore the quota (e.g. `python -m app.cli set-quota`) and settle it later.
    """


@dataclass(frozen=True)
class Reservation:
    """Plain values only, so nothing reads the session after the reservation commits."""

    reservation_id: uuid.UUID
    request_id: uuid.UUID
    application_id: uuid.UUID
    reserved_tokens: int


@dataclass(frozen=True)
class Settlement:
    status: ReservationStatus
    charged_tokens: int
    already_settled: bool


def estimate_request_tokens(messages: Sequence[ChatMessage], max_output_tokens: int) -> int:
    """Conservative upper estimate of a request's tokens. An estimate, never exact."""
    prompt = sum(len(m.content.encode("utf-8")) + MESSAGE_OVERHEAD_TOKENS for m in messages)
    return prompt + REQUEST_OVERHEAD_TOKENS + max_output_tokens


def reserve(
    session: Session,
    *,
    application_id: uuid.UUID,
    api_key_id: uuid.UUID,
    request_id: uuid.UUID,
    tokens: int,
    provider: str,
    model: str,
) -> Reservation:
    """Hold `tokens` of the application's quota, or raise InsufficientQuotaError."""
    if tokens <= 0:
        raise ValueError("tokens must be > 0")
    try:
        quota = session.scalar(
            select(TokenQuota).where(TokenQuota.application_id == application_id).with_for_update()
        )
        if quota is None:
            session.rollback()
            raise InsufficientQuotaError(tokens, 0, "no token quota is configured for this application")
        available = quota.available_tokens
        if tokens > available:
            session.rollback()  # releases the row lock
            raise InsufficientQuotaError(tokens, available, "token quota exceeded")

        reservation = QuotaReservation(
            application_id=application_id,
            api_key_id=api_key_id,
            request_id=request_id,
            provider=provider,
            model=model,
            reserved_tokens=tokens,
        )
        session.add(reservation)
        quota.reserved_tokens += tokens
        quota.updated_at = func.now()
        session.flush()
        result = Reservation(reservation.id, request_id, application_id, tokens)
        session.commit()
        return result
    except SQLAlchemyError:
        session.rollback()
        raise


def settle(
    session: Session,
    reservation_id: uuid.UUID,
    *,
    usage: Usage | None,
    succeeded: bool,
    may_have_consumed_tokens: bool,
    model: str | None = None,
    latency_ms: float | None = None,
    expired: bool = False,
) -> Settlement:
    """Settle a reservation exactly once. Calling it again for the same reservation is a no-op."""
    try:
        application_id = session.scalar(
            select(QuotaReservation.application_id).where(QuotaReservation.id == reservation_id)
        )
        if application_id is None:
            session.rollback()
            raise LookupError(f"no reservation {reservation_id}")
        quota = session.scalar(
            select(TokenQuota).where(TokenQuota.application_id == application_id).with_for_update()
        )
        if quota is None:
            # Without the quota row the counters can't be updated, so settle nothing at all:
            # releasing or charging here would lose track of the reserved tokens.
            session.rollback()
            raise QuotaNotFoundError(
                f"no token quota for application {application_id}; reservation {reservation_id} left reserved"
            )
        reservation = session.scalar(
            select(QuotaReservation)
            .where(QuotaReservation.id == reservation_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if reservation.status is not ReservationStatus.RESERVED:
            result = Settlement(reservation.status, reservation.charged_tokens or 0, already_settled=True)
            session.rollback()
            return result

        if usage is not None:
            status, charged, actual = ReservationStatus.RECONCILED, usage.total_tokens, usage.total_tokens
        elif expired:
            status, charged, actual = ReservationStatus.EXPIRED, reservation.reserved_tokens, None
        elif succeeded or may_have_consumed_tokens:
            status, charged, actual = ReservationStatus.RECONCILED, reservation.reserved_tokens, None
        else:
            status, charged, actual = ReservationStatus.RELEASED, 0, None

        quota.reserved_tokens -= reservation.reserved_tokens
        quota.consumed_tokens += charged
        quota.updated_at = func.now()
        reservation.status = status
        reservation.charged_tokens = charged
        reservation.actual_tokens = actual
        reservation.settled_at = func.now()
        reservation.updated_at = func.now()

        if status is not ReservationStatus.RELEASED:
            session.add(
                UsageRecord(
                    reservation_id=reservation.id,
                    request_id=reservation.request_id,
                    application_id=reservation.application_id,
                    api_key_id=reservation.api_key_id,
                    provider=reservation.provider,
                    model=model or reservation.model,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                    total_tokens=usage.total_tokens if usage else None,
                    usage_reported=usage is not None,
                    charged_tokens=charged,
                    status=UsageStatus.EXPIRED if expired else UsageStatus.SUCCEEDED if succeeded else UsageStatus.FAILED,
                    latency_ms=latency_ms,
                )
            )
        session.commit()
        return Settlement(status, charged, already_settled=False)
    except SQLAlchemyError:
        session.rollback()
        raise


def expire_stale_reservations(session: Session, older_than: timedelta) -> list[uuid.UUID]:
    """Settle reservations still `reserved` after `older_than` as expired (charging them).

    Recovers requests whose process crashed, was cancelled, or couldn't reach the database
    after the provider call. Run it with an age well above the provider timeout, so it never
    races a request that is still in flight. Returns the expired reservations' request IDs.
    """
    cutoff = datetime.now(timezone.utc) - older_than
    stale = session.execute(
        select(QuotaReservation.id, QuotaReservation.request_id)
        .where(QuotaReservation.status == ReservationStatus.RESERVED, QuotaReservation.created_at < cutoff)
        .order_by(QuotaReservation.created_at)
    ).all()
    session.rollback()
    expired = []
    for reservation_id, request_id in stale:
        # settle() re-checks the status under lock, so one settled meanwhile is skipped.
        result = settle(
            session, reservation_id, usage=None, succeeded=False, may_have_consumed_tokens=True, expired=True
        )
        if not result.already_settled:
            expired.append(request_id)
    return expired


def restore_missing_quota(session: Session, application_id: uuid.UUID, limit_tokens: int) -> TokenQuota:
    """Recreate a deleted quota row, rebuilding its counters from the existing reservations.

    Safe without extra locks: while the row is missing, reserve() and settle() can't change
    this application's reservations (both lock the quota row first and stop if it's absent),
    so the history read here can't change before the new row exists. If two restorations
    race, the primary key (application_id) lets only one INSERT succeed; the other raises
    IntegrityError. The caller commits.
    """
    still_reserved = (
        select(func.coalesce(func.sum(QuotaReservation.reserved_tokens), 0))
        .where(
            QuotaReservation.application_id == application_id,
            QuotaReservation.status == ReservationStatus.RESERVED,
        )
        .scalar_subquery()
    )
    already_charged = (
        select(func.coalesce(func.sum(QuotaReservation.charged_tokens), 0))
        .where(
            QuotaReservation.application_id == application_id,
            QuotaReservation.status != ReservationStatus.RESERVED,
        )
        .scalar_subquery()
    )
    # One statement, so both sums come from the same snapshot.
    reserved_tokens, consumed_tokens = session.execute(select(still_reserved, already_charged)).one()

    quota = TokenQuota(
        application_id=application_id,
        limit_tokens=limit_tokens,
        reserved_tokens=reserved_tokens,
        consumed_tokens=consumed_tokens,
    )
    session.add(quota)
    session.flush()
    return quota


def quota_inconsistencies(session: Session, application_id: uuid.UUID) -> list[str]:
    """Differences between the quota counters and the reservations they summarize (empty = consistent).

    Everything is read in ONE statement. Under READ COMMITTED (PostgreSQL's default) each
    statement sees a single snapshot, but separate statements may not: a settlement committing
    between two queries would show up as a false inconsistency.
    """
    active_reservations = (
        select(func.coalesce(func.sum(QuotaReservation.reserved_tokens), 0))
        .where(
            QuotaReservation.application_id == application_id,
            QuotaReservation.status == ReservationStatus.RESERVED,
        )
        .scalar_subquery()
    )
    settled_charges = (
        select(func.coalesce(func.sum(QuotaReservation.charged_tokens), 0))
        .where(
            QuotaReservation.application_id == application_id,
            QuotaReservation.status != ReservationStatus.RESERVED,
        )
        .scalar_subquery()
    )
    usage_charges = (
        select(func.coalesce(func.sum(UsageRecord.charged_tokens), 0))
        .where(UsageRecord.application_id == application_id)
        .scalar_subquery()
    )
    row = session.execute(
        select(
            TokenQuota.reserved_tokens,
            TokenQuota.consumed_tokens,
            active_reservations,
            settled_charges,
            usage_charges,
        ).where(TokenQuota.application_id == application_id)
    ).one_or_none()
    if row is None:
        return ["no quota row"]

    reserved_counter, consumed_counter, reserved, consumed, charged_in_usage = row
    problems = []
    if reserved_counter != reserved:
        problems.append(f"reserved_tokens {reserved_counter} != active reservations {reserved}")
    if consumed_counter != consumed:
        problems.append(f"consumed_tokens {consumed_counter} != settled reservations {consumed}")
    if consumed != charged_in_usage:
        problems.append(f"settled reservations {consumed} != usage records {charged_in_usage}")
    return problems
