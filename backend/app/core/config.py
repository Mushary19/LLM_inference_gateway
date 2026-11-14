from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, resolved from this file so it loads regardless of the working directory.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str

    # Capacity limits. The admission controller enforces max_queued_requests;
    # the scheduler enforces max_active_requests (concurrent executions).
    max_active_requests: int = Field(default=3, gt=0)
    max_queued_requests: int = Field(default=5, gt=0)

    # Scheduler. Disable to run the API without executing requests (the test suite does this).
    scheduler_enabled: bool = True
    scheduler_poll_interval_seconds: float = Field(default=0.1, gt=0)
    simulated_inference_delay_seconds: float = Field(default=2.0, ge=0)

    # Simulated GPU memory (accounting only; nothing is allocated). KV cache gets whatever is
    # left after model weights and runtime memory: 16384 - 10240 - 1024 = 5120 MB = 320 x 16 MB blocks.
    gpu_total_memory_mb: int = Field(default=16384, gt=0)
    gpu_model_memory_mb: int = Field(default=10240, ge=0)
    gpu_runtime_memory_mb: int = Field(default=1024, ge=0)
    kv_block_size_mb: int = Field(default=16, gt=0)
    # Simplified stand-in for the real per-token KV size. 512 KiB/token is roughly the size
    # for a 7B fp16 model, so one 16 MB block holds 32 tokens.
    simulated_kv_bytes_per_token: int = Field(default=512 * 1024, gt=0)

    # Inference engine. "simulated" (default, used by tests) needs no model; "huggingface"
    # loads hf_model_id at startup, downloading it on first use.
    inference_engine: Literal["simulated", "huggingface"] = "simulated"
    hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    inference_device: Literal["cpu"] = "cpu"
    # Upper bound for a request's max_new_tokens (requests above it get HTTP 422).
    max_generation_tokens: int = Field(default=512, gt=0)
    # Simultaneous model.generate() calls in the Hugging Face engine.
    inference_max_concurrency: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def _check_gpu_memory(self) -> "Settings":
        if self.gpu_total_memory_mb <= self.gpu_model_memory_mb + self.gpu_runtime_memory_mb:
            raise ValueError("gpu_total_memory_mb must be greater than model + runtime memory")
        return self


settings = Settings()
