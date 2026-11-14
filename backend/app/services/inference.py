"""Inference engine interface, shared result types, and a SIMULATED implementation.

SimulatedInferenceEngine only sleeps and returns a fixed placeholder string. It is the
default engine and is used by the test suite; the real model-backed engine lives in
app/services/hf_inference.py and is selected with INFERENCE_ENGINE=huggingface.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from app.core.tokens import estimate_input_tokens


@dataclass(frozen=True)
class InferenceJob:
    """The data an engine needs to run one claimed request (detached from any DB session)."""

    request_id: uuid.UUID
    prompt: str
    max_new_tokens: int
    input_tokens: int


def tokens_per_second(generated_tokens: int, execution_seconds: float) -> float:
    """Generation throughput; 0.0 when no measurable time elapsed."""
    if execution_seconds <= 0:
        return 0.0
    return generated_tokens / execution_seconds


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int
    # Monotonic wall time of the generation call itself (excludes queueing and DB writes).
    execution_seconds: float

    @property
    def tokens_per_second(self) -> float:
        return tokens_per_second(self.generated_tokens, self.execution_seconds)


class InferenceEngine(Protocol):
    def count_input_tokens(self, prompt: str) -> int: ...

    async def generate(self, job: InferenceJob) -> GenerationResult: ...


class SimulatedInferenceError(Exception):
    """Raised by the simulated engine for requests configured to fail."""


class SimulatedInferenceEngine:
    """Simulation only: waits `delay_seconds`, then returns a placeholder. Not an LLM.

    Token counts are word counts, since there is no tokenizer.
    """

    def __init__(
        self, delay_seconds: float, fail_request_ids: frozenset[uuid.UUID] = frozenset()
    ) -> None:
        self.delay_seconds = delay_seconds
        self.fail_request_ids = fail_request_ids

    def count_input_tokens(self, prompt: str) -> int:
        return estimate_input_tokens(prompt)

    async def generate(self, job: InferenceJob) -> GenerationResult:
        start = time.monotonic()
        await asyncio.sleep(self.delay_seconds)
        if job.request_id in self.fail_request_ids:
            raise SimulatedInferenceError(f"Simulated failure for request {job.request_id}")
        text = f"[simulated inference result for request {job.request_id}]"
        return GenerationResult(
            text=text,
            generated_tokens=len(text.split()),
            execution_seconds=time.monotonic() - start,
        )
