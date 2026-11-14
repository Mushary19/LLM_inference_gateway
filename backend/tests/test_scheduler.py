"""Scheduler tests.

These use real committed rows: the scheduler opens its own sessions in worker threads, so it
cannot see the rollback-only transaction used by the API tests. Every row is deleted afterwards.
"""

import asyncio
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.core.config import settings
from app.database.connection import SessionLocal
from app.database.models import InferenceRequest, InferenceResult, RequestPriority, RequestStatus
from app.main import app
from app.services.inference import InferenceJob, SimulatedInferenceEngine
from app.services.resources import GpuMemoryManager, KvCacheSizer
from app.services.scheduler import Scheduler

HIGH, NORMAL, LOW = RequestPriority.HIGH, RequestPriority.NORMAL, RequestPriority.LOW
FINISHED = (RequestStatus.COMPLETED, RequestStatus.FAILED)


class RequestFactory:
    def __init__(self) -> None:
        self.ids: list[uuid.UUID] = []

    def create(
        self,
        priority: RequestPriority = NORMAL,
        age_seconds: float = 0,
        input_tokens: int = 2,
        max_new_tokens: int = 1,
    ) -> uuid.UUID:
        """Commit a QUEUED request whose created_at is `age_seconds` in the past."""
        with SessionLocal() as db:
            request = InferenceRequest(
                prompt="scheduler test",
                priority=priority,
                max_new_tokens=max_new_tokens,
                input_tokens=input_tokens,
                created_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
            )
            db.add(request)
            db.commit()
            self.ids.append(request.id)
            return request.id

    def track(self, request_id: uuid.UUID) -> None:
        self.ids.append(request_id)


@pytest.fixture
def requests():
    with SessionLocal() as db:
        if db.scalar(select(func.count()).select_from(InferenceRequest)):
            pytest.skip("scheduler tests need an empty inference_requests table")
    factory = RequestFactory()
    try:
        yield factory
    finally:
        with SessionLocal() as db:
            db.execute(delete(InferenceRequest).where(InferenceRequest.id.in_(factory.ids)))
            db.commit()


class RecordingEngine:
    """Wraps the simulated engine and records what the scheduler asked it to do."""

    def __init__(
        self,
        delay: float = 0.02,
        fail_ids: frozenset[uuid.UUID] = frozenset(),
        memory: GpuMemoryManager | None = None,
    ) -> None:
        self.inner = SimulatedInferenceEngine(delay, fail_ids)
        self.memory = memory
        self.started: list[uuid.UUID] = []
        self.current_active = 0
        self.max_observed_active = 0
        self.state_during_execution: dict[uuid.UUID, tuple] = {}
        self.blocks_during_execution: dict[uuid.UUID, list[int]] = {}

    async def generate(self, job: InferenceJob) -> str:
        self.started.append(job.request_id)
        self.current_active += 1
        self.max_observed_active = max(self.max_observed_active, self.current_active)
        if self.memory is not None:
            self.blocks_during_execution[job.request_id] = self.memory.blocks_of(job.request_id)
        try:
            self.state_during_execution[job.request_id] = await asyncio.to_thread(
                _state, job.request_id
            )
            return await self.inner.generate(job)
        finally:
            self.current_active -= 1


def _state(request_id: uuid.UUID) -> tuple:
    with SessionLocal() as db:
        row = db.get(InferenceRequest, request_id)
        return row.status, row.started_at, row.completed_at


def _all_finished(ids: list[uuid.UUID]) -> Callable[[], bool]:
    def check() -> bool:
        with SessionLocal() as db:
            done = db.scalar(
                select(func.count())
                .select_from(InferenceRequest)
                .where(InferenceRequest.id.in_(ids), InferenceRequest.status.in_(FINISHED))
            )
        return done == len(ids)

    return check


async def _run_until(schedulers: list[Scheduler], condition: Callable[[], bool], timeout: float = 15):
    for scheduler in schedulers:
        scheduler.start()
    deadline = time.monotonic() + timeout
    try:
        while not await asyncio.to_thread(condition):
            assert time.monotonic() < deadline, "timed out waiting for the scheduler"
            await asyncio.sleep(0.02)
        # The loop must still be alive (e.g. after a failed request) right up to shutdown.
        assert all(scheduler.is_running for scheduler in schedulers)
    finally:
        for scheduler in schedulers:
            await scheduler.stop()


# Test memory: 4 MB blocks holding 4 tokens each (1 MiB/token), sized per test.
TEST_BLOCK_MB = 4
TEST_SIZER = KvCacheSizer(kv_bytes_per_token=1024 * 1024, block_size_mb=TEST_BLOCK_MB)


def make_memory(kv_blocks: int = 1000) -> GpuMemoryManager:
    return GpuMemoryManager(
        total_memory_mb=kv_blocks * TEST_BLOCK_MB + 100,
        model_memory_mb=80,
        runtime_memory_mb=20,
        kv_block_size_mb=TEST_BLOCK_MB,
    )


def make_scheduler(
    engine, max_active: int = 3, poll: float = 0.02, memory: GpuMemoryManager | None = None
) -> Scheduler:
    return Scheduler(
        SessionLocal,
        engine,
        max_active=max_active,
        poll_interval_seconds=poll,
        memory=memory if memory is not None else make_memory(),
        kv_sizer=TEST_SIZER,
    )


def run_scheduler(
    engine, condition, max_active: int = 3, poll: float = 0.02, memory: GpuMemoryManager | None = None
) -> Scheduler:
    scheduler = make_scheduler(engine, max_active=max_active, poll=poll, memory=memory)
    asyncio.run(_run_until([scheduler], condition))
    return scheduler


# --- Scheduling policy ---


def test_strict_priority_then_fifo_order(requests):
    a = requests.create(NORMAL, age_seconds=50)
    b = requests.create(HIGH, age_seconds=40)
    c = requests.create(LOW, age_seconds=30)
    d = requests.create(HIGH, age_seconds=20)
    e = requests.create(NORMAL, age_seconds=10)
    engine = RecordingEngine()

    run_scheduler(engine, _all_finished(requests.ids), max_active=1)

    assert engine.started == [b, d, a, e, c]


def test_fifo_within_same_priority(requests):
    # Inserted newest-first, so insertion order differs from created_at order.
    newest = requests.create(NORMAL, age_seconds=10)
    middle = requests.create(NORMAL, age_seconds=20)
    oldest = requests.create(NORMAL, age_seconds=30)
    engine = RecordingEngine()

    run_scheduler(engine, _all_finished(requests.ids), max_active=1)

    assert engine.started == [oldest, middle, newest]


def test_first_batch_claims_highest_priority_requests(requests):
    a = requests.create(NORMAL, age_seconds=50)
    b = requests.create(HIGH, age_seconds=40)
    c = requests.create(LOW, age_seconds=30)
    d = requests.create(HIGH, age_seconds=20)
    e = requests.create(NORMAL, age_seconds=10)
    engine = RecordingEngine(delay=0.2)

    run_scheduler(engine, _all_finished(requests.ids), max_active=3)

    assert engine.started[:3] == [b, d, a]
    assert set(engine.started[3:]) == {e, c}


# --- Capacity ---


def test_never_more_than_max_active_concurrent_executions(requests):
    for i in range(8):
        requests.create(NORMAL, age_seconds=10 - i)
    engine = RecordingEngine(delay=0.2)

    run_scheduler(engine, _all_finished(requests.ids), max_active=3)

    assert engine.max_observed_active == 3  # never above the limit, and the limit was used
    assert sorted(engine.started) == sorted(requests.ids)


# --- Lifecycle ---


def test_lifecycle_queued_running_completed(requests):
    request_id = requests.create()
    assert _state(request_id) == (RequestStatus.QUEUED, None, None)
    engine = RecordingEngine()

    run_scheduler(engine, _all_finished(requests.ids))

    status, started_at, completed_at = engine.state_during_execution[request_id]
    assert status is RequestStatus.RUNNING
    assert started_at is not None and completed_at is None

    status, started_at, completed_at = _state(request_id)
    assert status is RequestStatus.COMPLETED
    assert started_at is not None and completed_at is not None
    assert completed_at >= started_at


def test_failure_marks_failed_and_scheduler_continues(requests):
    failing = requests.create(HIGH, age_seconds=20)
    healthy = requests.create(NORMAL, age_seconds=10)
    engine = RecordingEngine(fail_ids=frozenset({failing}))

    run_scheduler(engine, _all_finished(requests.ids), max_active=1)

    assert engine.started == [failing, healthy]  # healthy ran after the failure
    assert engine.state_during_execution[failing][0] is RequestStatus.RUNNING
    status, started_at, completed_at = _state(failing)
    assert status is RequestStatus.FAILED
    assert started_at is not None and completed_at is not None
    assert _state(healthy)[0] is RequestStatus.COMPLETED


# --- Claim safety ---


def test_every_request_executes_exactly_once_with_competing_schedulers(requests):
    for i in range(12):
        requests.create(NORMAL, age_seconds=12 - i)
    engine = RecordingEngine(delay=0.05)
    schedulers = [
        make_scheduler(engine, max_active=3, poll=0.01) for _ in range(2)
    ]

    asyncio.run(_run_until(schedulers, _all_finished(requests.ids)))

    counts = Counter(engine.started)
    assert set(counts) == set(requests.ids)
    assert set(counts.values()) == {1}
    assert engine.max_observed_active <= 6  # 3 slots per scheduler


# --- Idle behaviour ---


def test_empty_queue_keeps_scheduler_alive_without_busy_spinning(requests):
    scheduler = make_scheduler(RecordingEngine(), max_active=3, poll=0.05)
    claims = 0
    original_claim = scheduler._claim

    def counting_claim(limit):
        nonlocal claims
        claims += 1
        return original_claim(limit)

    scheduler._claim = counting_claim

    async def idle_for_one_second():
        scheduler.start()
        cpu_before = time.process_time()
        await asyncio.sleep(1.0)
        cpu_used = time.process_time() - cpu_before
        alive = scheduler.is_running
        await scheduler.stop()
        return cpu_used, alive

    cpu_used, alive = asyncio.run(idle_for_one_second())

    assert alive
    assert 5 <= claims <= 30  # ~20 polls at 0.05s, not thousands
    assert cpu_used < 0.5


# --- Shutdown ---


def test_stop_cancels_in_flight_execution_and_leaves_it_running(requests):
    request_id = requests.create()
    scheduler = make_scheduler(RecordingEngine(delay=30), max_active=3, poll=0.02)

    async def start_then_stop():
        scheduler.start()
        while _state(request_id)[0] is not RequestStatus.RUNNING:
            await asyncio.sleep(0.02)
        stop_started = time.monotonic()
        await scheduler.stop()
        return time.monotonic() - stop_started

    stop_duration = asyncio.run(start_then_stop())

    assert stop_duration < 2  # did not wait for the 30s execution
    assert not scheduler.is_running and scheduler.active_count == 0
    # Documented behaviour: an interrupted request stays RUNNING until recovery exists.
    status, started_at, completed_at = _state(request_id)
    assert status is RequestStatus.RUNNING and completed_at is None


def test_app_lifespan_starts_and_stops_scheduler(requests, monkeypatch):
    monkeypatch.setattr(settings, "scheduler_enabled", True)
    monkeypatch.setattr(settings, "simulated_inference_delay_seconds", 0.05)
    monkeypatch.setattr(settings, "scheduler_poll_interval_seconds", 0.02)

    with TestClient(app) as client:
        scheduler = app.state.scheduler
        assert scheduler is not None and scheduler.is_running

        response = client.post("/generate", json={"prompt": "lifespan test", "max_new_tokens": 5})
        assert response.status_code == 202
        request_id = uuid.UUID(response.json()["request_id"])
        requests.track(request_id)

        deadline = time.monotonic() + 10
        while _state(request_id)[0] is not RequestStatus.COMPLETED:
            assert time.monotonic() < deadline, "request was not completed by the app's scheduler"
            time.sleep(0.02)

    assert scheduler._task.done() and scheduler._task.exception() is None
    assert not scheduler.is_running and scheduler.active_count == 0
    assert scheduler.memory.get_usage().kv_used_blocks == 0


def test_scheduler_not_started_when_disabled(client):
    assert app.state.scheduler is None


# --- Memory-aware scheduling ---
# TEST_SIZER: 4 tokens per block, so blocks = ceil((input_tokens + max_new_tokens) / 4).


def test_request_that_does_not_fit_waits_while_smaller_ones_run(requests):
    memory = make_memory(kv_blocks=10)
    occupant = uuid.uuid4()
    memory.allocate(occupant, 6)  # something else holds 6 blocks, 4 are free
    big = requests.create(HIGH, age_seconds=30, input_tokens=2, max_new_tokens=30)  # 8 blocks
    small_1 = requests.create(NORMAL, age_seconds=20, input_tokens=2, max_new_tokens=6)  # 2 blocks
    small_2 = requests.create(NORMAL, age_seconds=10, input_tokens=2, max_new_tokens=6)  # 2 blocks
    engine = RecordingEngine(memory=memory)
    scheduler = make_scheduler(engine, max_active=3, memory=memory)

    async def scenario():
        scheduler.start()
        try:
            while not await asyncio.to_thread(_all_finished([small_1, small_2])):
                await asyncio.sleep(0.02)
            # The highest-priority request is still waiting for memory, despite free slots.
            big_status_while_blocked = (await asyncio.to_thread(_state, big))[0]
            started_while_blocked = list(engine.started)
            memory.release(occupant)  # free the 6 blocks
            while not await asyncio.to_thread(_all_finished([big])):
                await asyncio.sleep(0.02)
            return big_status_while_blocked, started_while_blocked
        finally:
            await scheduler.stop()

    big_status_while_blocked, started_while_blocked = asyncio.run(scenario())

    assert big_status_while_blocked is RequestStatus.QUEUED
    assert started_while_blocked == [small_1, small_2]
    assert engine.started == [small_1, small_2, big]
    assert len(engine.blocks_during_execution[big]) == 8
    assert _state(big)[0] is RequestStatus.COMPLETED
    assert memory.get_usage().kv_used_blocks == 0


def test_request_larger_than_entire_kv_cache_fails_without_running(requests):
    memory = make_memory(kv_blocks=10)
    oversized = requests.create(HIGH, age_seconds=20, input_tokens=2, max_new_tokens=78)  # 20 blocks
    small = requests.create(NORMAL, age_seconds=10, input_tokens=2, max_new_tokens=6)
    engine = RecordingEngine(memory=memory)

    run_scheduler(engine, _all_finished(requests.ids), memory=memory)

    status, started_at, completed_at = _state(oversized)
    assert status is RequestStatus.FAILED
    assert started_at is None and completed_at is not None
    assert oversized not in engine.started
    assert _state(small)[0] is RequestStatus.COMPLETED
    assert memory.get_usage().kv_used_blocks == 0


def test_memory_limits_concurrency_below_execution_slots(requests):
    memory = make_memory(kv_blocks=10)
    for i in range(4):
        requests.create(NORMAL, age_seconds=10 - i, input_tokens=2, max_new_tokens=18)  # 5 blocks
    engine = RecordingEngine(delay=0.2, memory=memory)

    run_scheduler(engine, _all_finished(requests.ids), max_active=3, memory=memory)

    assert engine.max_observed_active == 2  # 3 slots, but only 2 x 5 blocks fit in 10
    assert sorted(engine.started) == sorted(requests.ids)
    assert memory.get_usage().kv_used_blocks == 0


def test_blocks_are_held_during_execution_and_released_after_success(requests):
    memory = make_memory(kv_blocks=10)
    request_id = requests.create(input_tokens=2, max_new_tokens=10)  # 12 tokens -> 3 blocks
    engine = RecordingEngine(memory=memory)

    run_scheduler(engine, _all_finished(requests.ids), memory=memory)

    assert len(engine.blocks_during_execution[request_id]) == 3
    assert _state(request_id)[0] is RequestStatus.COMPLETED
    usage = memory.get_usage()
    assert (usage.kv_used_blocks, usage.kv_free_blocks, usage.active_allocations) == (0, 10, 0)


def test_blocks_are_released_after_failure(requests):
    memory = make_memory(kv_blocks=10)
    failing = requests.create(input_tokens=2, max_new_tokens=10)
    engine = RecordingEngine(fail_ids=frozenset({failing}), memory=memory)

    run_scheduler(engine, _all_finished(requests.ids), memory=memory)

    assert len(engine.blocks_during_execution[failing]) == 3
    assert _state(failing)[0] is RequestStatus.FAILED
    usage = memory.get_usage()
    assert (usage.kv_used_blocks, usage.active_allocations) == (0, 0)


def test_blocks_are_released_when_stopped_mid_execution(requests):
    memory = make_memory(kv_blocks=10)
    request_id = requests.create(input_tokens=2, max_new_tokens=10)
    scheduler = make_scheduler(RecordingEngine(delay=30), memory=memory)

    async def start_then_stop():
        scheduler.start()
        while _state(request_id)[0] is not RequestStatus.RUNNING:
            await asyncio.sleep(0.02)
        held = memory.get_usage().kv_used_blocks
        await scheduler.stop()
        return held

    assert asyncio.run(start_then_stop()) == 3
    assert memory.get_usage().kv_used_blocks == 0


def test_failed_claim_transaction_releases_blocks_and_keeps_request_queued(requests):
    from sqlalchemy.exc import OperationalError

    memory = make_memory(kv_blocks=10)
    request_id = requests.create(input_tokens=2, max_new_tokens=10)

    def failing_session():
        session = SessionLocal()

        def fail():
            raise OperationalError("COMMIT", {}, Exception("simulated commit failure"))

        session.commit = fail
        return session

    scheduler = Scheduler(
        failing_session,
        RecordingEngine(),
        max_active=3,
        poll_interval_seconds=0.02,
        memory=memory,
        kv_sizer=TEST_SIZER,
    )

    with pytest.raises(OperationalError):
        scheduler._claim(3)

    assert memory.get_usage().kv_used_blocks == 0
    assert _state(request_id)[0] is RequestStatus.QUEUED


def test_app_lifespan_creates_gpu_memory_manager_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "scheduler_enabled", True)

    with TestClient(app):
        usage = app.state.gpu_memory.get_usage()
        assert app.state.scheduler.memory is app.state.gpu_memory

    assert usage.kv_total_blocks == 320 and usage.kv_used_blocks == 0



# --- Result persistence ---


def _results(request_id: uuid.UUID) -> list[InferenceResult]:
    with SessionLocal() as db:
        return list(db.scalars(select(InferenceResult).where(InferenceResult.request_id == request_id)))


def test_completed_request_persists_its_result_exactly_once(requests):
    request_id = requests.create(input_tokens=2, max_new_tokens=10)
    engine = RecordingEngine()
    scheduler = run_scheduler(engine, _all_finished(requests.ids))

    results = _results(request_id)
    assert len(results) == 1
    assert results[0].generated_text == f"[simulated inference result for request {request_id}]"
    assert results[0].generated_tokens == len(results[0].generated_text.split())
    assert results[0].expires_at > results[0].created_at

    # A second finish for the same request is refused by the RUNNING guard: no duplicate result.
    from app.services.inference import GenerationResult, InferenceJob

    duplicate = GenerationResult("again", generated_tokens=1, execution_seconds=0.1)
    scheduler._finish(InferenceJob(request_id, "x", 10, 2), RequestStatus.COMPLETED, duplicate)
    assert [r.generated_text for r in _results(request_id)] == [results[0].generated_text]


def test_failed_request_has_no_result(requests):
    failing = requests.create(input_tokens=2, max_new_tokens=10)
    healthy = requests.create(input_tokens=2, max_new_tokens=10)
    engine = RecordingEngine(fail_ids=frozenset({failing}))

    run_scheduler(engine, _all_finished(requests.ids))

    assert _state(failing)[0] is RequestStatus.FAILED
    assert _results(failing) == []
    assert len(_results(healthy)) == 1
