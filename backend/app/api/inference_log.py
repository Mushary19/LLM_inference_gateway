"""The `inference_request_finished` log line: one per chat completion that reached the
provider stage (after its quota reservation), on every path including failures and
cancellation.

Requests rejected before that (401, 429 rate limit or quota, 404 unknown model, 422, 503)
never start inference; they are covered by their own log lines and `http_request_finished`.

The outcome is what happened to the inference, which for a stream is not the HTTP status:
a stream answers 200 as soon as the provider accepts it and can still fail afterwards.

Outcomes:
- completed: the provider finished (finish reason stop or length).
- provider_error: the provider failed or broke its contract.
- timeout: the provider did not respond or finish in time.
- provider_rejected: the provider refused the request as invalid (an upstream 400/413/422, e.g.
  context too long). Gateway rejections (401, 422, 429, ...) happen before inference and never
  get an outcome.
- gateway_error: a bug or unexpected failure in the gateway.
- client_disconnected: the client went away before the stream finished.
- cancelled: the request's task was cancelled without a client disconnect (e.g. server shutdown).
"""

import logging
import time
from typing import Literal

import anyio

from app.core.structured_logging import log_event
from app.providers.base import (
    InvalidProviderRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    Usage,
)

logger = logging.getLogger(__name__)

EVENT = "inference_request_finished"

Outcome = Literal[
    "completed", "provider_error", "timeout", "provider_rejected", "gateway_error", "client_disconnected", "cancelled"
]
# not_attempted: no settlement was tried (the reservation stays for expire-reservations);
# unknown: the settlement call was interrupted, so whether it committed isn't known here.
Settlement = Literal["succeeded", "failed", "not_attempted", "unknown"]

_LEVELS: dict[str, int] = {"provider_error": logging.WARNING, "timeout": logging.WARNING, "gateway_error": logging.ERROR}


def provider_error_outcome(exc: ProviderError) -> Outcome:
    if isinstance(exc, ProviderTimeoutError):
        return "timeout"
    if isinstance(exc, InvalidProviderRequestError):
        return "provider_rejected"
    return "provider_error"


def log_provider_failure(
    *, provider: str, model: str, provider_model_id: str, stream: bool, phase: str, exc: ProviderError
) -> None:
    """The `provider_request_failed` line: structured metadata only.

    The exception's message is deliberately not logged: it carries upstream text (error
    messages, response fragments, URLs) that the gateway can't vouch for. The error type and
    the upstream HTTP status classify the failure; `phase` is "response" (non-streaming),
    "before_accept" or "mid_stream".
    """
    fields: dict[str, object] = {
        "provider": provider,
        "model": model,
        "stream": stream,
        "phase": phase,
        "error_type": type(exc).__name__,
        "upstream_status": exc.status_code,
        "outcome": provider_error_outcome(exc),
    }
    if isinstance(exc, ModelNotFoundError):
        # The registry routed a public model to a provider that doesn't serve it: a gateway
        # configuration problem. The provider model ID comes from the registry, not upstream.
        log_event(
            logger, logging.ERROR, "provider_request_failed",
            f"Provider {provider} does not serve model {provider_model_id!r} (public model {model}); "
            "check the model registry",
            provider_model_id=provider_model_id, **fields,
        )
        return
    status = f", upstream status {exc.status_code}" if exc.status_code is not None else ""
    log_event(
        logger, logging.WARNING, "provider_request_failed",
        f"Provider {provider} failed for model {model}: {type(exc).__name__}{status}",
        **fields,
    )


def interrupted_outcome(exc: BaseException) -> Outcome:
    """For an exception that isn't a ProviderError: cancellation or a gateway failure."""
    return "cancelled" if isinstance(exc, anyio.get_cancelled_exc_class()) else "gateway_error"


class InferenceSummary:
    """Collects one inference request's outcome and logs it exactly once (later calls to
    finish() do nothing). It never raises and never affects the request."""

    def __init__(self, *, provider: str, model: str, stream: bool, started: float) -> None:
        self.provider = provider
        self.model = model
        self.stream = stream
        self.started = started
        self.settlement: Settlement = "not_attempted"
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def finish(
        self,
        *,
        outcome: Outcome,
        http_status: int | None,
        usage: Usage | None,
        client_disconnected: bool = False,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        fields: dict[str, object] = {
            "provider": self.provider,
            "model": self.model,
            "stream": self.stream,
            "outcome": outcome,
            "http_status": http_status,
            "duration_ms": round((time.monotonic() - self.started) * 1000, 1),
            "usage_reported": usage is not None,
            "settlement": self.settlement,
        }
        if usage is not None:
            fields["prompt_tokens"] = usage.prompt_tokens
            fields["completion_tokens"] = usage.completion_tokens
        if self.stream:
            fields["stream_completed"] = outcome == "completed"
            fields["client_disconnected"] = client_disconnected
        log_event(
            logger,
            _LEVELS.get(outcome, logging.INFO),
            EVENT,
            f"Inference request finished: {outcome}",
            **fields,
        )

    def finish_interrupted(self, exc: BaseException) -> None:
        """For an exception that escaped before finish() was called (no-op otherwise)."""
        outcome = interrupted_outcome(exc)
        self.finish(outcome=outcome, http_status=500 if outcome == "gateway_error" else None, usage=None)
