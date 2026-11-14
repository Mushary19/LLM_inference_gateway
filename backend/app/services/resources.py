"""Simulated GPU memory accounting and KV-cache block allocation.

SIMULATION ONLY: no real memory is allocated and no tensors exist. The manager keeps
integer block IDs and an owner map so the scheduler can make memory-aware decisions.

GPU memory is split into:
- model weights   (reserved, fixed)
- runtime memory  (reserved, fixed: activations, temporary buffers)
- KV cache        (everything left over, divided into fixed-size blocks)

Requests own whole blocks. Blocks don't have to be contiguous, so a request can use any
free blocks and memory doesn't fragment into unusable gaps.
"""

import math
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass

BYTES_PER_MB = 1024 * 1024


class InsufficientKvCacheError(Exception):
    """Raised when a request needs more KV blocks than are currently free."""


class KvCacheSizer:
    """Turns a request's token count into a number of KV-cache blocks.

    NOT the real transformer formula. Real KV size depends on layers, KV heads, head
    dimension and dtype (roughly 2 * layers * kv_heads * head_dim * dtype_bytes per token).
    Here it is collapsed into a single configurable `kv_bytes_per_token`:

        blocks = ceil((input_tokens + max_new_tokens) * kv_bytes_per_token / block_bytes)

    It reserves room for the full output up front, because the scheduler allocates
    before generation starts and does not grow allocations yet.
    """

    def __init__(self, kv_bytes_per_token: int, block_size_mb: int) -> None:
        if kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be > 0")
        if block_size_mb <= 0:
            raise ValueError("block_size_mb must be > 0")
        self.kv_bytes_per_token = kv_bytes_per_token
        self.block_bytes = block_size_mb * BYTES_PER_MB

    def blocks_required(self, input_tokens: int, max_new_tokens: int) -> int:
        if input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        total_tokens = input_tokens + max_new_tokens
        return math.ceil(total_tokens * self.kv_bytes_per_token / self.block_bytes)


@dataclass(frozen=True)
class GpuMemoryUsage:
    total_memory_mb: int
    model_memory_mb: int
    runtime_memory_mb: int
    kv_cache_memory_mb: int
    kv_block_size_mb: int
    kv_total_blocks: int
    kv_used_blocks: int
    kv_free_blocks: int
    active_allocations: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class GpuMemoryManager:
    """Process-local accounting of a simulated GPU's KV-cache blocks.

    Thread-safe: allocations happen in the scheduler's DB worker thread and releases on the
    event loop, so every read or write of block state takes `_lock`.
    """

    def __init__(
        self,
        total_memory_mb: int,
        model_memory_mb: int,
        runtime_memory_mb: int,
        kv_block_size_mb: int,
    ) -> None:
        if model_memory_mb < 0 or runtime_memory_mb < 0:
            raise ValueError("model and runtime memory must be >= 0")
        if kv_block_size_mb <= 0:
            raise ValueError("kv_block_size_mb must be > 0")
        kv_cache_memory_mb = total_memory_mb - model_memory_mb - runtime_memory_mb
        if kv_cache_memory_mb <= 0:
            raise ValueError("total memory must be greater than model memory + runtime memory")
        # Whole blocks only; any remainder smaller than one block is unusable.
        kv_total_blocks = kv_cache_memory_mb // kv_block_size_mb
        if kv_total_blocks == 0:
            raise ValueError("KV cache memory is smaller than one block")

        self.total_memory_mb = total_memory_mb
        self.model_memory_mb = model_memory_mb
        self.runtime_memory_mb = runtime_memory_mb
        self.kv_cache_memory_mb = kv_cache_memory_mb
        self.kv_block_size_mb = kv_block_size_mb
        self.kv_total_blocks = kv_total_blocks

        self._lock = threading.Lock()
        self._free_blocks: deque[int] = deque(range(kv_total_blocks))
        self._owners: dict[uuid.UUID, list[int]] = {}

    def fits_at_all(self, num_blocks: int) -> bool:
        """Whether a request this size could ever run, even with an empty KV cache."""
        return 0 < num_blocks <= self.kv_total_blocks

    def can_allocate(self, num_blocks: int) -> bool:
        """Point-in-time check. Use allocate() for an atomic check-and-take."""
        with self._lock:
            return 0 < num_blocks <= len(self._free_blocks)

    def allocate(self, request_id: uuid.UUID, num_blocks: int) -> list[int]:
        """Atomically take `num_blocks` free blocks for `request_id`, or raise and take none."""
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        with self._lock:
            if request_id in self._owners:
                raise ValueError(f"request {request_id} already owns KV blocks")
            if num_blocks > len(self._free_blocks):
                raise InsufficientKvCacheError(
                    f"need {num_blocks} KV blocks, {len(self._free_blocks)} free"
                )
            blocks = [self._free_blocks.popleft() for _ in range(num_blocks)]
            self._owners[request_id] = blocks
            return list(blocks)

    def release(self, request_id: uuid.UUID) -> list[int]:
        """Free all blocks owned by `request_id`. Idempotent: unknown IDs release nothing."""
        with self._lock:
            blocks = self._owners.pop(request_id, [])
            self._free_blocks.extend(blocks)
            return list(blocks)

    def blocks_of(self, request_id: uuid.UUID) -> list[int]:
        with self._lock:
            return list(self._owners.get(request_id, []))

    def get_usage(self) -> GpuMemoryUsage:
        with self._lock:
            free = len(self._free_blocks)
            active = len(self._owners)
        return GpuMemoryUsage(
            total_memory_mb=self.total_memory_mb,
            model_memory_mb=self.model_memory_mb,
            runtime_memory_mb=self.runtime_memory_mb,
            kv_cache_memory_mb=self.kv_cache_memory_mb,
            kv_block_size_mb=self.kv_block_size_mb,
            kv_total_blocks=self.kv_total_blocks,
            kv_used_blocks=self.kv_total_blocks - free,
            kv_free_blocks=free,
            active_allocations=active,
        )
