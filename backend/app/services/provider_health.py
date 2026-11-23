"""On-demand provider health and model availability checks (python -m app.cli check-providers).

Three different questions, answered separately:
- Application health: is the API process up? That's GET /health, which checks nothing here.
- Provider health: can we reach this provider instance and does it accept our key?
- Model availability: does that provider list the upstream model ID a route uses?

Each provider is probed with ONE request: list_models(), which for OpenAI-compatible servers
is GET {base_url}/models. It is free (no inference, no tokens), and its answer serves both
questions. A provider can be healthy while one of its models is missing, and an unhealthy
provider tells us nothing about its models, so they are "unknown" rather than missing.

"healthy" only means the provider is reachable and authenticated. Inference can still fail
(e.g. token quota exhausted upstream), and "not_listed" doesn't prove a model unusable:
listings can omit aliases (Ollama serves "qwen2.5" but lists "qwen2.5:latest").

Reports contain fixed codes only. No upstream messages, URLs or credentials.
Nothing here is cached, scheduled or used for routing.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal

from app.providers.base import (
    ModelNotFoundError,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.registry import ModelRegistry, ModelRoute

logger = logging.getLogger(__name__)

ProviderStatus = Literal["healthy", "unhealthy", "unverified"]
Availability = Literal["listed", "not_listed", "unknown"]

DEFAULT_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ProviderCheck:
    name: str
    status: ProviderStatus
    # A fixed code (see classify_failure), None when healthy.
    reason: str | None
    latency_ms: float
    # Upstream model IDs from the listing; empty unless healthy. Not part of the report output.
    listed_models: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ModelCheck:
    public_id: str
    provider: str
    provider_model_id: str
    enabled: bool
    availability: Availability


@dataclass(frozen=True)
class HealthReport:
    providers: tuple[ProviderCheck, ...]
    models: tuple[ModelCheck, ...]

    @property
    def problems(self) -> list[str]:
        found = [
            f"provider {check.name} is {check.status} ({check.reason})"
            for check in self.providers
            if check.status != "healthy"
        ]
        found += [
            f"enabled model {check.public_id} is {check.availability} on provider {check.provider}"
            for check in self.models
            if check.enabled and check.availability != "listed"
        ]
        return found

    @property
    def ok(self) -> bool:
        """Every provider healthy and every enabled model listed."""
        return not self.problems


def classify_failure(exc: BaseException) -> tuple[ProviderStatus, str]:
    """Map a probe failure to (status, reason). Reasons are fixed codes, safe to show anywhere."""
    if isinstance(exc, (asyncio.TimeoutError, ProviderTimeoutError)):
        return "unhealthy", "timeout"
    if not isinstance(exc, ProviderError):
        return "unhealthy", "probe_error"  # a bug or an unexpected library error
    status_code = exc.status_code
    if status_code is None:
        if isinstance(exc, ProviderUnavailableError):
            return "unhealthy", "connection_failed"  # DNS, TLS, refused, reset
        return "unhealthy", "invalid_response"  # answered, but not a readable model list
    if status_code in (401, 403):
        return "unhealthy", "authentication_failed"
    if status_code == 404 or isinstance(exc, ModelNotFoundError):
        # The server answered but has no model listing: we can't verify it either way.
        return "unverified", "listing_not_supported"
    if status_code == 429:
        return "unhealthy", "rate_limited"
    if status_code >= 500:
        return "unhealthy", "server_error"
    if status_code == 400:
        return "unhealthy", "request_rejected"
    return "unhealthy", "unexpected_status"


async def probe_provider(name: str, provider: Provider, timeout: float) -> ProviderCheck:
    """Probe one provider. Never raises (except cancellation of the caller), so one broken
    provider can't stop the others from being checked.

    `timeout` bounds this probe only, independently of the provider's inference timeout. On
    expiry wait_for cancels the request task, which aborts the HTTP request and releases
    its connection.
    """
    started = time.monotonic()
    try:
        models = await asyncio.wait_for(provider.list_models(), timeout)
        listed = frozenset(model.id for model in models)
    except Exception as exc:  # deliberately broad: any failure is a result, not a crash
        status, reason = classify_failure(exc)
        # Metadata only: the exception's text is upstream content (messages, URLs). A traceback
        # is kept for unexpected (non-provider) exceptions, where it points at a gateway bug.
        logger.debug(
            "Probe of provider %s failed: %s (upstream status %s, reason %s)",
            name, type(exc).__name__, getattr(exc, "status_code", None), reason,
            exc_info=not isinstance(exc, ProviderError),
        )
        return ProviderCheck(name, status, reason, _elapsed_ms(started))
    return ProviderCheck(name, "healthy", None, _elapsed_ms(started), listed)


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 1)


def _model_check(route: ModelRoute, enabled: bool, providers: dict[int, ProviderCheck]) -> ModelCheck:
    check = providers[id(route.provider)]
    if check.status != "healthy":
        availability: Availability = "unknown"
    elif route.provider_model_id in check.listed_models:
        availability = "listed"
    else:
        availability = "not_listed"
    return ModelCheck(route.public_id, check.name, route.provider_model_id, enabled, availability)


async def check_registry(registry: ModelRegistry, timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS) -> HealthReport:
    """Probe every provider concurrently (once each), then derive every model's availability.

    Takes about `timeout` at most, however many providers there are. Never calls chat.
    """
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    names = sorted(registry.providers)
    checks = await asyncio.gather(
        *(probe_provider(name, registry.providers[name], timeout) for name in names)
    )
    by_provider = {id(registry.providers[name]): check for name, check in zip(names, checks)}
    models = [_model_check(route, True, by_provider) for route in registry.enabled_models()]
    models += [_model_check(route, False, by_provider) for route in registry.disabled_models()]
    models.sort(key=lambda check: check.public_id)
    return HealthReport(providers=tuple(checks), models=tuple(models))
