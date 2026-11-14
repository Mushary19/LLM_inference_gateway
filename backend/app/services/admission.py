"""Admission control: decides whether a new request may enter the system.

This answers only "should this request be accepted?". Choosing which accepted
request runs next is the scheduler's job and does not live here.
"""

import enum
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import InferenceRequest, RequestStatus

# Arbitrary fixed key for the PostgreSQL advisory lock that serializes admission decisions.
ADMISSION_LOCK_KEY = 7_301_001


class AdmissionDecision(str, enum.Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class AdmissionResult:
    decision: AdmissionDecision
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.decision is AdmissionDecision.ACCEPT


class AdmissionController:
    def __init__(self, max_queued_requests: int, max_active_requests: int) -> None:
        self.max_queued_requests = max_queued_requests
        # Not part of the decision: concurrent executions are limited by the scheduler, and
        # admission only bounds how many requests may wait.
        self.max_active_requests = max_active_requests

    def decide(self, queued_count: int) -> AdmissionResult:
        """Pure policy: current queue depth -> decision."""
        if queued_count >= self.max_queued_requests:
            return AdmissionResult(
                AdmissionDecision.REJECT,
                f"Queue capacity reached ({queued_count}/{self.max_queued_requests} queued)",
            )
        return AdmissionResult(AdmissionDecision.ACCEPT)

    def evaluate(self, db: Session) -> AdmissionResult:
        """Count queued requests and decide.

        Concurrency: a plain COUNT-then-INSERT lets two concurrent requests both see 4 queued
        and both be accepted. To prevent that, this takes a transaction-scoped advisory lock
        before counting. The lock is held until the caller's transaction ends, so the caller
        MUST insert (if accepted) and commit, or roll back (if rejected), in the same transaction.

        Limitations: this serializes all admission decisions (fine at this scale), and it only
        guards writers that take the same lock, which today is only this method.
        """
        db.execute(select(func.pg_advisory_xact_lock(ADMISSION_LOCK_KEY)))
        queued_count = db.scalar(
            select(func.count())
            .select_from(InferenceRequest)
            .where(InferenceRequest.status == RequestStatus.QUEUED)
        )
        return self.decide(queued_count)
