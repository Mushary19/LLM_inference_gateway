import threading
import uuid

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.resources import GpuMemoryManager, InsufficientKvCacheError, KvCacheSizer


@pytest.fixture
def memory() -> GpuMemoryManager:
    # Defaults from the spec: 16384 - 10240 - 1024 = 5120 MB of KV cache = 320 x 16 MB blocks.
    return GpuMemoryManager(
        total_memory_mb=16384, model_memory_mb=10240, runtime_memory_mb=1024, kv_block_size_mb=16
    )


# --- Initialization ---


def test_kv_capacity_is_whatever_model_and_runtime_leave(memory):
    usage = memory.get_usage()

    assert usage.kv_cache_memory_mb == 5120
    assert usage.kv_total_blocks == 320
    assert usage.to_dict() == {
        "total_memory_mb": 16384,
        "model_memory_mb": 10240,
        "runtime_memory_mb": 1024,
        "kv_cache_memory_mb": 5120,
        "kv_block_size_mb": 16,
        "kv_total_blocks": 320,
        "kv_used_blocks": 0,
        "kv_free_blocks": 320,
        "active_allocations": 0,
    }


def test_partial_block_at_the_end_is_not_usable():
    memory = GpuMemoryManager(
        total_memory_mb=100, model_memory_mb=50, runtime_memory_mb=10, kv_block_size_mb=16
    )

    assert memory.kv_cache_memory_mb == 40
    assert memory.kv_total_blocks == 2  # 32 MB usable, 8 MB remainder


@pytest.mark.parametrize(
    "total, model, runtime, block",
    [
        (16384, 10240, 6144, 16),  # total == model + runtime: no KV cache
        (8000, 10240, 1024, 16),  # total < model + runtime
        (16384, 10240, 1024, 0),  # zero block size
        (16384, 10240, 1024, -16),  # negative block size
        (16384, -1, 1024, 16),  # negative model memory
        (100, 50, 40, 16),  # KV cache (10 MB) smaller than one block
    ],
)
def test_invalid_memory_configuration_is_rejected(total, model, runtime, block):
    with pytest.raises(ValueError):
        GpuMemoryManager(
            total_memory_mb=total,
            model_memory_mb=model,
            runtime_memory_mb=runtime,
            kv_block_size_mb=block,
        )


def test_settings_reject_total_not_greater_than_model_plus_runtime():
    with pytest.raises(ValidationError, match="greater than model"):
        Settings(
            database_url="postgresql+psycopg://x",
            gpu_total_memory_mb=11264,
            gpu_model_memory_mb=10240,
            gpu_runtime_memory_mb=1024,
        )


@pytest.mark.parametrize(
    "field", ["kv_block_size_mb", "simulated_kv_bytes_per_token", "gpu_total_memory_mb"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_settings_reject_non_positive_sizes(field, value):
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql+psycopg://x", **{field: value})


def test_default_settings_are_valid():
    settings = Settings(database_url="postgresql+psycopg://x")

    assert (settings.gpu_total_memory_mb, settings.gpu_model_memory_mb) == (16384, 10240)
    assert (settings.gpu_runtime_memory_mb, settings.kv_block_size_mb) == (1024, 16)
    assert settings.simulated_kv_bytes_per_token == 512 * 1024


# --- KV-cache sizing ---


def test_sizer_rounds_up_to_whole_blocks():
    sizer = KvCacheSizer(kv_bytes_per_token=512 * 1024, block_size_mb=16)  # 32 tokens per block

    assert sizer.blocks_required(input_tokens=0, max_new_tokens=1) == 1
    assert sizer.blocks_required(input_tokens=2, max_new_tokens=30) == 1  # exactly 32 tokens
    assert sizer.blocks_required(input_tokens=2, max_new_tokens=31) == 2  # 33 tokens
    assert sizer.blocks_required(input_tokens=5, max_new_tokens=100) == 4  # 105 tokens


def test_more_tokens_never_need_fewer_blocks():
    sizer = KvCacheSizer(kv_bytes_per_token=512 * 1024, block_size_mb=16)
    blocks = [sizer.blocks_required(10, n) for n in range(1, 500)]

    assert blocks == sorted(blocks)
    assert blocks[0] < blocks[-1]


@pytest.mark.parametrize("input_tokens, max_new_tokens", [(0, 0), (5, 0), (5, -1), (-1, 5)])
def test_sizer_rejects_invalid_token_counts(input_tokens, max_new_tokens):
    with pytest.raises(ValueError):
        KvCacheSizer(512 * 1024, 16).blocks_required(input_tokens, max_new_tokens)


@pytest.mark.parametrize("bytes_per_token, block_mb", [(0, 16), (-1, 16), (1024, 0)])
def test_sizer_rejects_invalid_configuration(bytes_per_token, block_mb):
    with pytest.raises(ValueError):
        KvCacheSizer(bytes_per_token, block_mb)


# --- Allocation ---


def test_allocate_one_request(memory):
    request = uuid.uuid4()

    blocks = memory.allocate(request, 3)

    assert len(blocks) == 3 and len(set(blocks)) == 3
    assert all(0 <= b < 320 for b in blocks)
    assert memory.blocks_of(request) == blocks
    usage = memory.get_usage()
    assert (usage.kv_used_blocks, usage.kv_free_blocks, usage.active_allocations) == (3, 317, 1)


def test_multiple_requests_get_disjoint_blocks(memory):
    sizes = {uuid.uuid4(): n for n in (3, 2, 7, 1)}

    owned = {request: memory.allocate(request, n) for request, n in sizes.items()}

    for request, n in sizes.items():
        assert len(owned[request]) == n
        assert memory.blocks_of(request) == owned[request]
    all_blocks = [b for blocks in owned.values() for b in blocks]
    assert len(all_blocks) == len(set(all_blocks)) == 13
    assert memory.get_usage().kv_used_blocks == 13


def test_same_request_cannot_allocate_twice(memory):
    request = uuid.uuid4()
    memory.allocate(request, 2)

    with pytest.raises(ValueError):
        memory.allocate(request, 2)
    assert memory.get_usage().kv_used_blocks == 2


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_non_positive_allocation_is_rejected(memory, num_blocks):
    with pytest.raises(ValueError):
        memory.allocate(uuid.uuid4(), num_blocks)
    assert not memory.can_allocate(num_blocks)


def test_exhausted_cache_rejects_further_allocation(memory):
    memory.allocate(uuid.uuid4(), 300)
    memory.allocate(uuid.uuid4(), 20)

    assert memory.get_usage().kv_free_blocks == 0
    assert not memory.can_allocate(1)
    blocked = uuid.uuid4()
    with pytest.raises(InsufficientKvCacheError):
        memory.allocate(blocked, 1)
    assert memory.blocks_of(blocked) == []
    assert memory.get_usage().active_allocations == 2


def test_failed_allocation_takes_nothing(memory):
    memory.allocate(uuid.uuid4(), 315)

    with pytest.raises(InsufficientKvCacheError):
        memory.allocate(uuid.uuid4(), 6)  # 5 free: no partial allocation

    assert memory.get_usage().kv_free_blocks == 5
    assert memory.can_allocate(5)


def test_oversized_request_fails_cleanly(memory):
    request = uuid.uuid4()

    assert not memory.fits_at_all(321)
    assert memory.fits_at_all(320)
    with pytest.raises(InsufficientKvCacheError):
        memory.allocate(request, 321)
    assert memory.get_usage().kv_free_blocks == 320
    assert memory.blocks_of(request) == []


# --- Release ---


def test_release_returns_blocks_to_the_free_pool(memory):
    first, second = uuid.uuid4(), uuid.uuid4()
    first_blocks = memory.allocate(first, 300)
    memory.allocate(second, 20)
    assert not memory.can_allocate(1)

    released = memory.release(first)

    assert released == first_blocks
    assert memory.blocks_of(first) == []
    assert memory.get_usage().kv_free_blocks == 300
    # The freed blocks are reusable, and never handed to two owners at once.
    third_blocks = memory.allocate(uuid.uuid4(), 300)
    assert set(third_blocks) == set(first_blocks)
    assert not set(third_blocks) & set(memory.blocks_of(second))


def test_release_is_idempotent(memory):
    request = uuid.uuid4()
    memory.allocate(request, 4)

    assert len(memory.release(request)) == 4
    assert memory.release(request) == []
    assert memory.release(uuid.uuid4()) == []
    assert memory.get_usage().kv_free_blocks == 320


# --- Concurrency ---


def test_concurrent_allocations_never_exceed_capacity(memory):
    threads_count, blocks_each = 32, 25  # 800 blocks requested, 320 available
    barrier = threading.Barrier(threads_count)
    granted: dict[uuid.UUID, list[int]] = {}
    refused = 0
    results_lock = threading.Lock()

    def worker():
        nonlocal refused
        request = uuid.uuid4()
        barrier.wait()
        try:
            blocks = memory.allocate(request, blocks_each)
        except InsufficientKvCacheError:
            with results_lock:
                refused += 1
        else:
            with results_lock:
                granted[request] = blocks

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(granted) == 320 // blocks_each  # 12 fit, 20 refused
    assert refused == threads_count - len(granted)
    all_blocks = [b for blocks in granted.values() for b in blocks]
    assert len(all_blocks) == len(set(all_blocks))
    assert memory.get_usage().kv_used_blocks == len(all_blocks) == 300

    # Concurrent releases restore every block exactly once.
    release_threads = [threading.Thread(target=memory.release, args=(r,)) for r in granted]
    for thread in release_threads:
        thread.start()
    for thread in release_threads:
        thread.join()
    usage = memory.get_usage()
    assert (usage.kv_free_blocks, usage.active_allocations) == (320, 0)


def test_concurrent_allocate_and_release_keep_accounting_consistent(memory):
    errors: list[Exception] = []

    def churn():
        try:
            for _ in range(200):
                request = uuid.uuid4()
                try:
                    blocks = memory.allocate(request, 7)
                except InsufficientKvCacheError:
                    continue
                assert len(blocks) == 7
                memory.release(request)
        except Exception as exc:  # surfaced in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=churn) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    usage = memory.get_usage()
    assert (usage.kv_used_blocks, usage.kv_free_blocks, usage.active_allocations) == (0, 320, 0)
