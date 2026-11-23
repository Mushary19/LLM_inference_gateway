"""Prometheus metrics: one registry per application instance.

GatewayMetrics owns a CollectorRegistry of its own, never prometheus-client's global default
one. So every application instance (and every test that installs a fresh GatewayMetrics)
has independent metrics, and defining the same metric names again never collides.

Metric definitions belong in GatewayMetrics.__init__, as attributes, registered with
`registry=self.registry`. Instrumentation reaches the current application's instance via
`metrics_of(app)` (from a request: `request.app`, or `scope["app"]` in ASGI middleware), so
a test can swap it out per test.

Scope (Milestone 7B.1): the registry and the /metrics endpoint only; no metrics are defined
yet. Values are kept in this process: a deployment with several worker processes would
need prometheus-client's multiprocess mode, which isn't set up (the gateway runs one
uvicorn process).
"""

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from starlette.types import ASGIApp

# The route serving the metrics; excluded from HTTP request metrics once those exist, so
# scrapes don't count as traffic.
METRICS_PATH = "/metrics"
CONTENT_TYPE = CONTENT_TYPE_LATEST


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)

    def exposition(self) -> bytes:
        """The registry in Prometheus text format. Reads in-memory values only."""
        return generate_latest(self.registry)


def metrics_of(app: ASGIApp) -> GatewayMetrics:
    return app.state.metrics  # type: ignore[attr-defined]
