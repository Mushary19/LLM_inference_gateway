"""Real CPU inference with a Hugging Face causal language model.

Loaded once per process at application startup (HuggingFaceInferenceEngine.load), then
shared by the scheduler (generation) and the API (input token counting). FP32 on CPU, no
quantization, no batching, no streaming: this is the unoptimized baseline.
"""

import asyncio
import logging
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.services.inference import GenerationResult, InferenceJob

logger = logging.getLogger(__name__)


class InferenceGenerationError(Exception):
    """Raised when the model fails to generate a response for a request."""


class HuggingFaceInferenceEngine:
    def __init__(
        self,
        tokenizer,
        model,
        device: str,
        max_new_tokens_limit: int,
        max_concurrency: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_new_tokens_limit = max_new_tokens_limit
        # Fast tokenizers are not safe to call from several threads at once, and the API
        # (token counting) and the scheduler (generation) both use this one instance.
        self._tokenizer_lock = threading.Lock()
        # Caps simultaneous model.generate() calls. Concurrent CPU generation competes for the
        # same cores and memory bandwidth, so more than 1 usually makes every request slower.
        self._generation_slots = threading.BoundedSemaphore(max_concurrency)

    @classmethod
    def load(
        cls, model_id: str, device: str, max_new_tokens_limit: int, max_concurrency: int
    ) -> "HuggingFaceInferenceEngine":
        """Load tokenizer and model (downloading them on first use). Blocking and slow."""
        if device != "cpu":
            raise ValueError(f"only CPU inference is supported, got device={device!r}")
        logger.info("Loading %s on %s (FP32)", model_id, device)
        start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        model.to(device)
        model.eval()
        logger.info("Loaded %s in %.1fs", model_id, time.monotonic() - start)
        return cls(tokenizer, model, device, max_new_tokens_limit, max_concurrency)

    def _chat_inputs(self, prompt: str):
        """Tokenize a single-turn user prompt with the model's chat template.

        Used for both counting and generation, so stored input_tokens match what the model sees.
        """
        messages = [{"role": "user", "content": prompt}]
        with self._tokenizer_lock:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

    def count_input_tokens(self, prompt: str) -> int:
        return int(self._chat_inputs(prompt)["input_ids"].shape[-1])

    async def generate(self, job: InferenceJob) -> GenerationResult:
        # model.generate() is blocking and CPU-bound; keep it off the event loop.
        return await asyncio.to_thread(self._generate_blocking, job)

    def _generate_blocking(self, job: InferenceJob) -> GenerationResult:
        try:
            inputs = self._chat_inputs(job.prompt).to(self.device)
            prompt_length = inputs["input_ids"].shape[-1]
            max_new_tokens = min(job.max_new_tokens, self.max_new_tokens_limit)
            with self._generation_slots:
                start = time.monotonic()
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                execution_seconds = time.monotonic() - start
            # generate() returns prompt + continuation; decode only the new tokens.
            new_token_ids = output_ids[0, prompt_length:]
            with self._tokenizer_lock:
                text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        except Exception as exc:
            raise InferenceGenerationError(f"generation failed for request {job.request_id}") from exc
        return GenerationResult(
            text=text.strip(),
            generated_tokens=int(new_token_ids.shape[-1]),
            execution_seconds=execution_seconds,
        )
