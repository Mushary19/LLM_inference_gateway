import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.types import ASGIApp

from app.api import chat, usage  # noqa: F401  (register their routes on the /v1 router)
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.middleware import RequestContextMiddleware
from app.api.routes import router
from app.api.v1 import router as v1_router
from app.core.config import settings
from app.core.metrics import GatewayMetrics
from app.core.structured_logging import configure_logging
from app.database.connection import SessionLocal
from app.providers.registry import registry_from_settings
from app.services.inference import InferenceEngine, SimulatedInferenceEngine
from app.services.rate_limit import TokenBucketRateLimiter, create_redis_client
from app.services.resources import GpuMemoryManager, KvCacheSizer
from app.services.scheduler import Scheduler

# At import, so after uvicorn has applied its own logging configuration (it does that before
# importing the app) and before anything below logs.
configure_logging(settings.log_level, settings.log_format)

logger = logging.getLogger(__name__)


def create_rate_limiter() -> TokenBucketRateLimiter:
    """One Redis client and connection pool for the process, shared by all requests."""
    redis = create_redis_client(settings.redis_url, settings.redis_timeout_seconds)
    return TokenBucketRateLimiter(
        redis,
        requests_per_minute=settings.rate_limit_requests_per_minute,
        burst=settings.rate_limit_burst,
    )


async def create_inference_engine() -> InferenceEngine:
    if settings.inference_engine == "huggingface":
        # Imported lazily so the simulated engine doesn't pay for importing torch.
        from app.services.hf_inference import HuggingFaceInferenceEngine

        # Loading (and the first-time download) blocks for a while; keep it off the event loop.
        return await asyncio.to_thread(
            HuggingFaceInferenceEngine.load,
            model_id=settings.hf_model_id,
            device=settings.inference_device,
            max_new_tokens_limit=settings.max_generation_tokens,
            max_concurrency=settings.inference_max_concurrency,
        )
    return SimulatedInferenceEngine(settings.simulated_inference_delay_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Gateway path (/v1/chat/completions, /v1/models). Built first: an invalid registry
    # configuration stops startup before anything else is started.
    model_registry = registry_from_settings(settings)
    try:
        app.state.model_registry = model_registry

        # Legacy path (/generate): one engine per process, used for token counting and by the scheduler.
        inference_engine = await create_inference_engine()
        app.state.inference_engine = inference_engine
        scheduler = None
        gpu_memory = None
        if settings.scheduler_enabled:
            gpu_memory = GpuMemoryManager(
                total_memory_mb=settings.gpu_total_memory_mb,
                model_memory_mb=settings.gpu_model_memory_mb,
                runtime_memory_mb=settings.gpu_runtime_memory_mb,
                kv_block_size_mb=settings.kv_block_size_mb,
            )
            scheduler = Scheduler(
                session_factory=SessionLocal,
                engine=inference_engine,
                max_active=settings.max_active_requests,
                poll_interval_seconds=settings.scheduler_poll_interval_seconds,
                memory=gpu_memory,
                kv_sizer=KvCacheSizer(settings.simulated_kv_bytes_per_token, settings.kv_block_size_mb),
            )
            scheduler.start()
        app.state.scheduler = scheduler
        app.state.gpu_memory = gpu_memory

        rate_limiter = create_rate_limiter()
        app.state.rate_limiter = rate_limiter
        try:
            yield
        finally:
            try:
                if scheduler is not None:
                    await scheduler.stop()
            finally:
                try:
                    await rate_limiter.aclose()
                except Exception:
                    logger.exception("Failed to close the Redis rate-limit client")
    finally:
        await model_registry.aclose()  # closes every provider; logs, never raises, on failure


class InferenceFlowApp(FastAPI):
    """FastAPI with RequestContextMiddleware around its entire middleware stack.

    app.add_middleware() would place it inside ServerErrorMiddleware, which sends the 500
    for an unhandled exception straight to the server, so that response would lack
    X-Request-ID. Outermost, every response, that 500 included, passes through it.

    Each instance also gets its own metrics registry (app.state.metrics, see
    app/core/metrics.py), so two instances in one process never share or collide.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state.metrics = GatewayMetrics()

    def build_middleware_stack(self) -> ASGIApp:
        return RequestContextMiddleware(super().build_middleware_stack())


app = InferenceFlowApp(title="InferenceFlow", lifespan=lifespan)
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(router)
app.include_router(v1_router)
