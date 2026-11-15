"""SQLAlchemy ORM models.

Define models here as subclasses of ``Base``. Alembic imports this module,
so any model defined (or imported) here is picked up by autogenerate.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class RequestPriority(enum.IntEnum):
    LOW = 1
    NORMAL = 2
    HIGH = 3


class RequestStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


_STATUS_VALUES = ", ".join(f"'{status.value}'" for status in RequestStatus)


class InferenceRequest(Base):
    __tablename__ = "inference_requests"
    __table_args__ = (
        CheckConstraint("priority BETWEEN 1 AND 3", name="ck_inference_requests_priority_range"),
        CheckConstraint("max_new_tokens > 0", name="ck_inference_requests_max_new_tokens_positive"),
        CheckConstraint("input_tokens >= 0", name="ck_inference_requests_input_tokens_non_negative"),
        CheckConstraint(f"status IN ({_STATUS_VALUES})", name="ck_inference_requests_status_valid"),
        # Matches the scheduler query: WHERE status = ? ORDER BY priority DESC, created_at ASC.
        Index(
            "ix_inference_requests_status_priority_created",
            "status",
            text("priority DESC"),
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    # Stored as a plain integer; RequestPriority is an IntEnum, so it compares equal to its int value.
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, default=RequestPriority.NORMAL, server_default=text("2")
    )
    max_new_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    # native_enum=False stores the enum as VARCHAR(20) rather than a PostgreSQL ENUM type.
    status: Mapped[RequestStatus] = mapped_column(
        Enum(RequestStatus, native_enum=False, length=20),
        nullable=False,
        default=RequestStatus.QUEUED,
        server_default=RequestStatus.QUEUED.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # One-to-one: a request has at most one result. Deleting a result never touches the request.
    result: Mapped["InferenceResult | None"] = relationship(
        back_populates="request", cascade="all, delete-orphan", passive_deletes=True
    )
    # One-to-one: a request has at most one metrics record. No delete cascade in either direction:
    # passive_deletes="all" leaves the row alone, so the foreign key blocks deleting a request with metrics.
    metrics: Mapped["RequestMetrics | None"] = relationship(
        back_populates="request", passive_deletes="all"
    )


class InferenceResult(Base):
    __tablename__ = "inference_results"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_inference_results_request_id"),
        CheckConstraint(
            "generated_tokens >= 0", name="ck_inference_results_generated_tokens_non_negative"
        ),
        CheckConstraint("expires_at > created_at", name="ck_inference_results_expires_after_created"),
        Index("ix_inference_results_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "inference_requests.id",
            name="fk_inference_results_request_id_inference_requests",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    generated_text: Mapped[str] = mapped_column(Text, nullable=False)
    generated_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Results are retained for 30 minutes; after that they are eligible for cleanup (not yet implemented).
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now() + interval '30 minutes'")
    )

    request: Mapped[InferenceRequest] = relationship(back_populates="result")


class RequestMetrics(Base):
    """Latency measurements for one request, in milliseconds.

    queue_wait_ms           created_at -> started_at
    time_to_first_token_ms  created_at -> first generated token (includes queue wait)
    execution_time_ms       started_at -> generation finished
    total_latency_ms        created_at -> generation finished (= queue_wait + execution_time)
    """

    __tablename__ = "request_metrics"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_request_metrics_request_id"),
        CheckConstraint("queue_wait_ms >= 0", name="ck_request_metrics_queue_wait_ms_non_negative"),
        CheckConstraint(
            "time_to_first_token_ms >= 0", name="ck_request_metrics_time_to_first_token_ms_non_negative"
        ),
        CheckConstraint(
            "execution_time_ms >= 0", name="ck_request_metrics_execution_time_ms_non_negative"
        ),
        CheckConstraint("total_latency_ms >= 0", name="ck_request_metrics_total_latency_ms_non_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_requests.id", name="fk_request_metrics_request_id_inference_requests"),
        nullable=False,
    )
    queue_wait_ms: Mapped[float] = mapped_column(Float, nullable=False)
    time_to_first_token_ms: Mapped[float] = mapped_column(Float, nullable=False)
    execution_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    request: Mapped[InferenceRequest] = relationship(back_populates="metrics")


class ClientApplication(Base):
    """A client application that calls the gateway. API keys belong to an application."""

    __tablename__ = "client_applications"
    __table_args__ = (
        UniqueConstraint("name", name="uq_client_applications_name"),
        CheckConstraint("length(trim(name)) > 0", name="ck_client_applications_name_not_blank"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="application")


class ApiKey(Base):
    """An API key. Only the SHA-256 hash of the full key is stored, never the key itself.

    key_prefix (e.g. "if_k3F9xQ2m") is the non-secret part of the key, used to identify it
    in listings and revocation. Authentication looks keys up by key_hash only.
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        UniqueConstraint("key_prefix", name="uq_api_keys_key_prefix"),
        UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
        CheckConstraint("key_hash ~ '^[0-9a-f]{64}$'", name="ck_api_keys_key_hash_sha256_hex"),
        CheckConstraint("length(trim(name)) > 0", name="ck_api_keys_name_not_blank"),
        CheckConstraint(
            "(is_active AND revoked_at IS NULL) OR (NOT is_active AND revoked_at IS NOT NULL)",
            name="ck_api_keys_revocation_consistent",
        ),
        Index("ix_api_keys_application_id", "application_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )
    application_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("client_applications.id", name="fk_api_keys_application_id_client_applications"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(32), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    application: Mapped[ClientApplication] = relationship(back_populates="api_keys")
