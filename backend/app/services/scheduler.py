"""Scheduler: claims QUEUED requests and runs them, subject to two limits:

1. execution capacity: at most `max_active` concurrent executions
2. memory capacity: the request's KV-cache blocks must be free in the GpuMemoryManager

Policy: strict priority (priority DESC), FIFO within a priority (created_at ASC), but a
request that does not fit in the free KV cache is skipped (it stays QUEUED) so smaller
requests behind it can run. No head-of-line blocking.
Consequences: strict priority can starve LOW requests while higher-priority work keeps
arriving, and a large request can be starved by a stream of small ones; there is no aging
or reservation yet. Running requests are never preempted.
A request that needs more blocks than the entire KV cache can never run, so it is marked
FAILED instead of occupying a queue slot forever.

Concurrency model:
- The scheduler loop and request executions are asyncio tasks on the app's event loop.
- The project's SQLAlchemy sessions are synchronous, so every DB call runs in a worker
  thread via asyncio.to_thread and never blocks the event loop.
- Runtime capacity is owned in memory: `_active` holds one task per running execution, and
  the loop only claims as many requests as there are free slots. KV blocks are owned by the
  GpuMemoryManager and released in a `finally` when an execution ends for any reason.
- PostgreSQL holds persisted state. Claims use FOR UPDATE SKIP LOCKED, so several schedulers
  (or worker processes) could share the table without ever claiming the same request.

Shutdown: stop() ends the loop and cancels in-flight executions. Requests that were RUNNING
at that moment stay RUNNING in the database; recovering them is a later phase.
"""

import asyncio
import logging
from collections.abc import Callable

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.database.models import InferenceRequest, InferenceResult, RequestStatus
from app.services.inference import GenerationResult, InferenceEngine, InferenceJob
from app.services.resources import GpuMemoryManager, InsufficientKvCacheError, KvCacheSizer

logger = logging.getLogger(__name__)

# How many queued candidates one claim may lock and consider. Lets small requests behind a
# large one be found, while bounding the work per poll. The admission queue limit is far below this.
CLAIM_SCAN_LIMIT = 50


class Scheduler:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        engine: InferenceEngine,
        max_active: int,
        poll_interval_seconds: float,
        memory: GpuMemoryManager,
        kv_sizer: KvCacheSizer,
    ) -> None:
        self.session_factory = session_factory
        self.engine = engine
        self.max_active = max_active
        self.memory = memory
        self.kv_sizer = kv_sizer
        self.poll_interval_seconds = poll_interval_seconds
        self._active: set[asyncio.Task] = set()
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._wake = asyncio.Event()  # set when a slot frees up or stop() is called

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def active_count(self) -> int:
        return len(self._active)

    def start(self) -> None:
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_loop(), name="inference-scheduler")

    async def stop(self) -> None:
        self._stopping.set()
        self._wake.set()
        if self._task is not None:
            await self._task
        in_flight = list(self._active)
        for task in in_flight:
            task.cancel()
        await asyncio.gather(*in_flight, return_exceptions=True)
        if in_flight:
            logger.warning("Scheduler stopped with %d request(s) left RUNNING", len(in_flight))

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            free_slots = self.max_active - len(self._active)
            if free_slots > 0:
                try:
                    jobs = await asyncio.to_thread(self._claim, free_slots)
                except Exception:
                    logger.exception("Failed to claim queued requests")
                    jobs = []
                for job in jobs:
                    self._start_execution(job)
            await self._idle()

    async def _idle(self) -> None:
        """Wait until a slot (and its KV blocks) frees, stop() is called, or the poll interval elapses."""
        try:
            await asyncio.wait_for(self._wake.wait(), self.poll_interval_seconds)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()

    def _start_execution(self, job: InferenceJob) -> None:
        task = asyncio.create_task(self._execute(job), name=f"inference-{job.request_id}")
        self._active.add(task)
        task.add_done_callback(self._on_execution_done)

    def _on_execution_done(self, task: asyncio.Task) -> None:
        self._active.discard(task)
        self._wake.set()

    async def _execute(self, job: InferenceJob) -> None:
        try:
            output: GenerationResult | None = None
            try:
                output = await self.engine.generate(job)
            except Exception:
                logger.exception("Inference failed for request %s", job.request_id)
                status = RequestStatus.FAILED
            else:
                status = RequestStatus.COMPLETED
                logger.info(
                    "Request %s generated: input_tokens=%d generated_tokens=%d "
                    "execution=%.3fs tokens_per_second=%.2f",
                    job.request_id, job.input_tokens, output.generated_tokens,
                    output.execution_seconds, output.tokens_per_second,
                )
            try:
                await asyncio.to_thread(self._finish, job, status, output)
            except Exception:
                logger.exception("Failed to mark request %s as %s", job.request_id, status.value)
        finally:
            # Runs on success, failure and cancellation, so blocks can't leak.
            released = self.memory.release(job.request_id)
            logger.info("Released %d KV blocks from request %s", len(released), job.request_id)

    def _claim(self, free_slots: int) -> list[InferenceJob]:
        """Start up to `free_slots` QUEUED requests whose KV blocks fit, in scheduling order.

        One transaction:
        1. Lock up to CLAIM_SCAN_LIMIT queued rows in priority order (SKIP LOCKED, so rows
           another scheduler is considering are left alone).
        2. Walk them in order and try to allocate each one's KV blocks. A request that doesn't
           fit right now is skipped and stays QUEUED; one that can never fit is marked FAILED.
        3. UPDATE only the requests that got blocks to RUNNING, then commit.
        If anything fails before the commit, the blocks taken in step 2 are released and the
        transaction rolls back, so no row becomes RUNNING without memory (or vice versa).
        """
        allocated: list[InferenceJob] = []
        with self.session_factory() as db:
            try:
                candidates = db.execute(
                    select(
                        InferenceRequest.id,
                        InferenceRequest.prompt,
                        InferenceRequest.input_tokens,
                        InferenceRequest.max_new_tokens,
                    )
                    .where(InferenceRequest.status == RequestStatus.QUEUED)
                    .order_by(InferenceRequest.priority.desc(), InferenceRequest.created_at.asc())
                    .limit(CLAIM_SCAN_LIMIT)
                    .with_for_update(skip_locked=True)
                ).all()

                impossible: list = []
                for row in candidates:
                    if len(allocated) == free_slots:
                        break
                    blocks_needed = self.kv_sizer.blocks_required(row.input_tokens, row.max_new_tokens)
                    if not self.memory.fits_at_all(blocks_needed):
                        logger.warning(
                            "Request %s needs %d KV blocks but the cache only has %d; marking FAILED",
                            row.id, blocks_needed, self.memory.kv_total_blocks,
                        )
                        impossible.append(row.id)
                        continue
                    try:
                        blocks = self.memory.allocate(row.id, blocks_needed)
                    except InsufficientKvCacheError:
                        logger.debug("Request %s waits for %d KV blocks", row.id, blocks_needed)
                        continue
                    logger.info("Allocated %d KV blocks to request %s", len(blocks), row.id)
                    allocated.append(
                        InferenceJob(
                            request_id=row.id,
                            prompt=row.prompt,
                            max_new_tokens=row.max_new_tokens,
                            input_tokens=row.input_tokens,
                        )
                    )

                if allocated:
                    db.execute(
                        update(InferenceRequest)
                        .where(InferenceRequest.id.in_([job.request_id for job in allocated]))
                        .values(status=RequestStatus.RUNNING, started_at=func.now())
                        .execution_options(synchronize_session=False)
                    )
                if impossible:
                    db.execute(
                        update(InferenceRequest)
                        .where(InferenceRequest.id.in_(impossible))
                        .values(status=RequestStatus.FAILED, completed_at=func.now())
                        .execution_options(synchronize_session=False)
                    )
                db.commit()
            except BaseException:
                for job in allocated:
                    self.memory.release(job.request_id)
                raise
        return allocated

    def _finish(
        self, job: InferenceJob, status: RequestStatus, output: GenerationResult | None = None
    ) -> None:
        """Record the final status and, for COMPLETED, the generated result, in one transaction.

        The RUNNING -> final transition can only succeed once, and the result is inserted only
        when it does, so a result is persisted at most once (the unique request_id backs this up).
        """
        with self.session_factory() as db:
            transition = db.execute(
                update(InferenceRequest)
                .where(InferenceRequest.id == job.request_id)
                .where(InferenceRequest.status == RequestStatus.RUNNING)
                .values(status=status, completed_at=func.now())
                .execution_options(synchronize_session=False)
            )
            if transition.rowcount != 1:
                db.rollback()
                logger.warning(
                    "Request %s was not RUNNING when finishing as %s", job.request_id, status.value
                )
                return
            if status is RequestStatus.COMPLETED and output is not None:
                db.add(
                    InferenceResult(
                        request_id=job.request_id,
                        generated_text=output.text,
                        generated_tokens=output.generated_tokens,
                    )
                )
            db.commit()
