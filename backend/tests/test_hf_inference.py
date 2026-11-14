"""HuggingFaceInferenceEngine tests with fake tokenizer/model objects. Nothing is downloaded."""

import asyncio
import threading
import time
import uuid

import pytest
import torch
from transformers import BatchEncoding

from app.core.config import settings
from app.database.models import InferenceRequest
from app.main import app
from app.services import hf_inference
from app.services.hf_inference import HuggingFaceInferenceEngine, InferenceGenerationError
from app.services.inference import InferenceJob, SimulatedInferenceEngine, tokens_per_second

EOS = 1
# What the fake chat template wraps around the user's words, like Qwen's
# "<|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n".
TEMPLATE_PREFIX = [100, 101, 102]
TEMPLATE_SUFFIX = [103, 104, 105, 106]
TEMPLATE_OVERHEAD = len(TEMPLATE_PREFIX) + len(TEMPLATE_SUFFIX)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = EOS

    def __init__(self) -> None:
        self.template_calls: list[tuple[list[dict], dict]] = []
        self.decoded: list[list[int]] = []
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def token_id(self, word: str) -> int:
        if word not in self._ids:
            self._ids[word] = 1000 + len(self._ids)
            self._words[self._ids[word]] = word
        return self._ids[word]

    def apply_chat_template(self, conversation, **kwargs):
        self.template_calls.append((conversation, kwargs))
        words = conversation[-1]["content"].split()
        ids = torch.tensor([TEMPLATE_PREFIX + [self.token_id(w) for w in words] + TEMPLATE_SUFFIX])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        ids = token_ids.tolist()
        self.decoded.append(ids)
        return " ".join(
            self._words.get(i, f"<{i}>") for i in ids if not (skip_special_tokens and i < 1000)
        )


class FakeModel:
    """Echoes the prompt ids back followed by a fixed reply, like model.generate() does."""

    def __init__(self, tokenizer: FakeTokenizer, reply: str = "Paris is the capital.", delay: float = 0):
        self.reply_ids = [tokenizer.token_id(w) for w in reply.split()] + [EOS]
        self.delay = delay
        self.fail_with: Exception | None = None
        self.generate_calls: list[dict] = []
        self.generate_threads: list[int] = []
        self.inference_mode_flags: list[bool] = []
        self.moved_to: list[str] = []
        self.eval_called = False
        self.current = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def to(self, device: str) -> "FakeModel":
        self.moved_to.append(device)
        return self

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id):
        with self._lock:
            self.current += 1
            self.max_concurrent = max(self.max_concurrent, self.current)
        try:
            self.generate_calls.append(
                {"input_ids": input_ids, "max_new_tokens": max_new_tokens, "pad_token_id": pad_token_id}
            )
            self.generate_threads.append(threading.get_ident())
            self.inference_mode_flags.append(torch.is_inference_mode_enabled())
            if self.delay:
                time.sleep(self.delay)  # blocking, like real CPU generation
            if self.fail_with is not None:
                raise self.fail_with
            new = torch.tensor([self.reply_ids[:max_new_tokens]])
            return torch.cat([input_ids, new], dim=1)
        finally:
            with self._lock:
                self.current -= 1


def make_engine(delay: float = 0, limit: int = 512, concurrency: int = 1):
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, delay=delay)
    engine = HuggingFaceInferenceEngine(
        tokenizer, model, device="cpu", max_new_tokens_limit=limit, max_concurrency=concurrency
    )
    return engine, tokenizer, model


def job(prompt: str = "What is the capital of France?", max_new_tokens: int = 50) -> InferenceJob:
    return InferenceJob(uuid.uuid4(), prompt, max_new_tokens, input_tokens=0)


@pytest.fixture
def fake_pretrained(monkeypatch):
    """Replace from_pretrained with fakes and count how often each is called."""
    calls = {"tokenizer": [], "model": []}
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)

    def tokenizer_from_pretrained(model_id, **kwargs):
        calls["tokenizer"].append((model_id, kwargs))
        return tokenizer

    def model_from_pretrained(model_id, **kwargs):
        calls["model"].append((model_id, kwargs))
        return model

    monkeypatch.setattr(hf_inference.AutoTokenizer, "from_pretrained", tokenizer_from_pretrained)
    monkeypatch.setattr(hf_inference.AutoModelForCausalLM, "from_pretrained", model_from_pretrained)
    return calls, tokenizer, model


# --- Loading ---


def test_load_initializes_tokenizer_and_fp32_model_on_cpu_in_eval_mode(fake_pretrained):
    calls, tokenizer, model = fake_pretrained

    engine = HuggingFaceInferenceEngine.load(
        "Qwen/Qwen2.5-0.5B-Instruct", device="cpu", max_new_tokens_limit=512, max_concurrency=1
    )

    assert calls["tokenizer"] == [("Qwen/Qwen2.5-0.5B-Instruct", {})]
    assert calls["model"] == [("Qwen/Qwen2.5-0.5B-Instruct", {"dtype": torch.float32})]
    assert model.moved_to == ["cpu"] and model.eval_called
    assert engine.tokenizer is tokenizer and engine.model is model


def test_load_rejects_non_cpu_device(fake_pretrained):
    calls, _, _ = fake_pretrained

    with pytest.raises(ValueError, match="CPU"):
        HuggingFaceInferenceEngine.load("m", device="cuda", max_new_tokens_limit=512, max_concurrency=1)
    assert calls["model"] == []


def test_model_is_loaded_once_and_reused_across_requests(fake_pretrained):
    calls, _, model = fake_pretrained
    engine = HuggingFaceInferenceEngine.load("m", device="cpu", max_new_tokens_limit=512, max_concurrency=1)

    async def several_requests():
        for _ in range(3):
            engine.count_input_tokens("hello there")
            await engine.generate(job())

    asyncio.run(several_requests())

    assert len(calls["tokenizer"]) == 1 and len(calls["model"]) == 1
    assert len(model.generate_calls) == 3


# --- Chat template and decoding ---


def test_prompt_is_formatted_with_the_chat_template():
    engine, tokenizer, model = make_engine()

    asyncio.run(engine.generate(job("Explain KV caching")))

    conversation, kwargs = tokenizer.template_calls[0]
    assert conversation == [{"role": "user", "content": "Explain KV caching"}]
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is True and kwargs["return_dict"] is True
    sent = model.generate_calls[0]["input_ids"][0].tolist()
    assert sent[:3] == TEMPLATE_PREFIX and sent[-4:] == TEMPLATE_SUFFIX


def test_generated_text_excludes_the_prompt():
    engine, tokenizer, model = make_engine()

    result = asyncio.run(engine.generate(job("What is the capital of France?")))

    assert result.text == "Paris is the capital."
    assert "France" not in result.text
    # Only the new tokens were decoded: no prompt or template ids.
    assert tokenizer.decoded == [model.reply_ids]


def test_input_tokens_are_counted_with_the_chat_template():
    engine, tokenizer, model = make_engine()
    prompt = "What is the capital of France?"  # 6 words

    count = engine.count_input_tokens(prompt)

    assert count == 6 + TEMPLATE_OVERHEAD
    assert count != len(prompt.split())
    asyncio.run(engine.generate(job(prompt)))
    assert model.generate_calls[0]["input_ids"].shape[-1] == count  # same tokens the model saw


# --- Execution ---


def test_generation_runs_outside_the_event_loop():
    engine, _, model = make_engine(delay=0.3)
    ticks = 0

    async def scenario():
        nonlocal ticks
        loop_thread = threading.get_ident()
        generation = asyncio.create_task(engine.generate(job()))
        while not generation.done():
            ticks += 1
            await asyncio.sleep(0.02)
        await generation
        return loop_thread

    loop_thread = asyncio.run(scenario())

    assert model.generate_threads and model.generate_threads[0] != loop_thread
    assert ticks >= 5  # the loop kept running while the model blocked for 0.3s


def test_generation_uses_inference_mode():
    engine, _, model = make_engine()

    asyncio.run(engine.generate(job()))

    assert model.inference_mode_flags == [True]


def test_max_new_tokens_is_capped_by_the_configured_limit():
    engine, _, model = make_engine(limit=2)

    result = asyncio.run(engine.generate(job(max_new_tokens=50)))

    assert model.generate_calls[0]["max_new_tokens"] == 2
    assert result.generated_tokens == 2


def test_generated_token_count_and_throughput():
    engine, _, model = make_engine(delay=0.1)

    result = asyncio.run(engine.generate(job()))

    assert result.generated_tokens == len(model.reply_ids)  # 4 words + EOS
    assert result.execution_seconds >= 0.1
    assert result.tokens_per_second == pytest.approx(result.generated_tokens / result.execution_seconds)


@pytest.mark.parametrize("tokens, seconds, expected", [(10, 2.0, 5.0), (0, 1.0, 0.0), (5, 0.0, 0.0), (5, -1.0, 0.0)])
def test_tokens_per_second_handles_zero_duration(tokens, seconds, expected):
    assert tokens_per_second(tokens, seconds) == expected


def test_generation_errors_are_wrapped():
    engine, _, model = make_engine()
    model.fail_with = RuntimeError("out of memory")

    with pytest.raises(InferenceGenerationError) as excinfo:
        asyncio.run(engine.generate(job()))

    assert isinstance(excinfo.value.__cause__, RuntimeError)


@pytest.mark.parametrize("concurrency", [1, 2])
def test_concurrent_generate_calls_are_limited(concurrency):
    engine, _, model = make_engine(delay=0.1, concurrency=concurrency)

    async def three_at_once():
        await asyncio.gather(*(engine.generate(job()) for _ in range(3)))

    asyncio.run(three_at_once())

    assert model.max_concurrent == concurrency


# --- Application wiring ---


def test_app_uses_simulated_engine_by_default(client):
    assert isinstance(app.state.inference_engine, SimulatedInferenceEngine)


@pytest.fixture
def hf_engine_app(monkeypatch):
    """Make the app build a HuggingFaceInferenceEngine from fakes instead of downloading."""
    monkeypatch.setattr(settings, "inference_engine", "huggingface")
    loads = []
    engine, _, _ = make_engine()

    def fake_load(**kwargs):
        loads.append(kwargs)
        return engine

    monkeypatch.setattr(HuggingFaceInferenceEngine, "load", staticmethod(fake_load))
    return engine, loads


def test_app_loads_hf_engine_once_at_startup_and_counts_real_tokens(hf_engine_app, client, db_session):
    engine, loads = hf_engine_app

    assert app.state.inference_engine is engine
    assert loads == [
        {
            "model_id": settings.hf_model_id,
            "device": "cpu",
            "max_new_tokens_limit": settings.max_generation_tokens,
            "max_concurrency": settings.inference_max_concurrency,
        }
    ]

    response = client.post("/generate", json={"prompt": "Explain KV caching", "max_new_tokens": 20})
    client.post("/generate", json={"prompt": "Another prompt", "max_new_tokens": 20})

    assert response.status_code == 202
    stored = db_session.get(InferenceRequest, uuid.UUID(response.json()["request_id"]))
    assert stored.input_tokens == 3 + TEMPLATE_OVERHEAD  # chat-template tokens, not 3 words
    assert len(loads) == 1  # requests reuse the startup engine
