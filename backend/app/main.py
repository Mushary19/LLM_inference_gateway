import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, usage  # noqa: F401  (register their routes on the /v1 router)
from app.api.routes import router
from app.api.v1 import router as v1_router
from app.core.config import settings
from app.database.connection import SessionLocal
from app.providers.factory import create_provider
from app.services.inference import InferenceEngine, SimulatedInferenceEngine
from app.services.rate_limit import TokenBucketRateLimiter, create_redis_client
from app.services.resources import GpuMemoryManager, KvCacheSizer
from app.services.scheduler import Scheduler

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

    # Gateway path (/v1/chat/completions): calls the provider directly.
    api_key = settings.openai_compatible_api_key
    chat_provider = create_provider(
        settings.chat_provider,
        base_url=settings.openai_compatible_base_url,
        api_key=api_key.get_secret_value() if api_key else None,
        timeout_seconds=settings.openai_compatible_timeout_seconds,
    )
    app.state.chat_provider = chat_provider
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
                await chat_provider.aclose()
            finally:
                try:
                    await rate_limiter.aclose()
                except Exception:
                    logger.exception("Failed to close the Redis rate-limit client")


app = FastAPI(title="InferenceFlow", lifespan=lifespan)
app.include_router(router)
app.include_router(v1_router)
