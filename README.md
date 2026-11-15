# InferenceFlow

A simplified AI inference gateway and resource manager.

> Status: work in progress. Request API, admission control, scheduler, simulated GPU memory
> accounting and real CPU inference with a small Hugging Face model are implemented.

## Tech stack

**Backend:** Python, FastAPI, SQLAlchemy, Alembic, psycopg, Pydantic Settings
**Inference:** PyTorch (CPU-only build), Hugging Face Transformers
**Database:** PostgreSQL 17
**Infrastructure:** Docker, Docker Compose

## Getting started

```bash
# Start PostgreSQL
docker compose up -d

# Set up the backend
cd backend
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Layout

- `backend/` – FastAPI service
- `frontend/` – reserved for future development

## Resource management (simulated GPU memory and KV cache)

> **Simulation only.** No GPU memory is allocated and no KV tensors exist. The
> `GpuMemoryManager` (`backend/app/services/resources.py`) only keeps track of integer block
> IDs so the scheduler can make the same kind of decisions a real inference server makes.

### Execution capacity vs memory capacity

The scheduler enforces two separate limits, and a request runs only when **both** allow it:

| Limit | What it bounds | Setting |
|---|---|---|
| Execution capacity | how many requests run at the same time | `MAX_ACTIVE_REQUESTS` (3) |
| Memory capacity | how much KV cache the running requests hold in total | `GPU_*`, `KV_BLOCK_SIZE_MB` |

`max_active_requests` alone isn't enough for LLM inference, because requests differ hugely in
cost. A 100-token request and a 6,000-token request each take one execution slot, but the large
one needs about 50 times more KV-cache memory. With 3 slots but only enough KV memory for 2
large requests, at most 2 large requests can run at once. The third slot sits idle, or goes to
a smaller request that fits.

### Why GPU memory is a resource

A GPU's memory is fixed and shared by:

- **model weights:** loaded once and always resident (default 10,240 MB)
- **runtime memory:** activations and temporary buffers (default 1,024 MB)
- **KV cache:** everything left over (default 16,384 − 10,240 − 1,024 = 5,120 MB)

Weights and runtime memory are reserved up front, so the KV cache is the part that running
requests compete for. If the scheduler ignored it, a real server would run out of memory partway
through generating.

### What the simulated KV cache represents

During generation a transformer stores the attention keys and values of every token processed
so far, so it doesn't have to recompute them. That storage grows with the number of tokens. Here
a request reserves room for `input_tokens + max_new_tokens` up front:

```
blocks = ceil((input_tokens + max_new_tokens) * SIMULATED_KV_BYTES_PER_TOKEN / block_size)
```

This is **not** the real formula. Real KV size depends on the number of layers, KV heads, head
dimension and dtype. The default of 512 KiB per token is only roughly the size for a 7B fp16
model, so one 16 MB block holds 32 tokens.

### Why fixed-size blocks

The KV cache is split into fixed-size blocks (default 16 MB, so 320 blocks). A request owns a
list of block IDs that don't have to be next to each other, for example `[192..319, 0..59]`
after earlier requests have finished. If each request needed one contiguous region instead,
free memory would break into gaps that are individually too small to use, even when the total
free space would be enough. With blocks, any free block is usable. The cost is rounding each
request up to a whole block.

### Why a request can stay QUEUED while execution slots are free

On each poll, the scheduler goes through the queued requests in priority order and tries to
allocate each one's blocks. A request that doesn't fit right now is **skipped and left
QUEUED**, and smaller requests behind it can still run. There is no head-of-line blocking. Its
blocks are allocated in the same database transaction that marks it RUNNING, and they are
released in a `finally` when it completes, fails or is cancelled. A request that needs more
blocks than the entire KV cache can never run, so it is marked FAILED.

Trade-off: a large request can be overtaken repeatedly by smaller ones (memory starvation), in
the same way that strict priority can starve LOW requests. There is no aging or memory
reservation yet.

## Real CPU inference (Hugging Face)

InferenceFlow has two inference engines, selected with `INFERENCE_ENGINE`:

| Engine | What it does | Use it for |
|---|---|---|
| `simulated` (default) | sleeps, then returns a placeholder string | tests, scheduler experiments |
| `huggingface` | runs a real model on the CPU | actual text generation |

**Model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), a
chat model with about 494M parameters, loaded in **FP32 on CPU**. There is no quantization,
batching or streaming: this is the unoptimized baseline.

### Memory and first-time download

- **Download:** about 0.95 GB (`model.safetensors` is stored in BF16). The first startup
  downloads it into the Hugging Face cache (`~/.cache/huggingface`). Later startups load it
  from there in a few seconds and work offline.
- **RAM:** the weights are converted to FP32, about 494M × 4 bytes ≈ **1.9 GB**. Measured
  resident memory of the backend process is about **2.4 GB** after loading, and stays roughly
  the same while generating short responses.
- **Speed:** about **4–6 generated tokens/second** on an Intel i5 (8th gen, 4 cores) with one
  request at a time.

### Enable it and start the backend

```bash
docker compose up -d                    # PostgreSQL
cd backend
source env/bin/activate
pip install -r requirements.txt         # includes CPU-only torch and transformers
INFERENCE_ENGINE=huggingface MAX_ACTIVE_REQUESTS=1 uvicorn app.main:app
```

You can put `INFERENCE_ENGINE=huggingface` in `backend/.env` instead. Startup waits until the
model has loaded (and been downloaded, the first time).

`MAX_ACTIVE_REQUESTS=1` is recommended on a laptop CPU. Requests running at the same time
compete for the same cores, so running them together makes each one slower rather than
raising throughput. `INFERENCE_MAX_CONCURRENCY` (default 1) also limits how many
`model.generate()` calls run at once. With more scheduler slots than that, the extra requests
are `RUNNING` (holding a slot and KV blocks) while they wait for the model.

Other settings: `HF_MODEL_ID`, `INFERENCE_DEVICE` (only `cpu` is accepted), and
`MAX_GENERATION_TOKENS` (default 512; requests with a larger `max_new_tokens` get HTTP 422).

### Send a request

```bash
curl -X POST localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt": "What is the capital of France?", "max_new_tokens": 40}'
# {"request_id": "…", "status": "QUEUED"}
```

The prompt is sent as a single user message through the model's chat template. Qwen's template
adds a default system prompt ("You are Qwen, created by Alibaba Cloud…"), so even a short
prompt is about 35 input tokens. `input_tokens` is counted with that same template, using the
model's tokenizer.

### Retrieve the result

There is no status endpoint yet. Results are stored in PostgreSQL:

```bash
docker compose exec db psql -U inferenceflow -d inferenceflow -c "
  SELECT q.status, q.input_tokens, r.generated_tokens, r.generated_text,
         q.completed_at - q.started_at AS execution_time
  FROM inference_requests q LEFT JOIN inference_results r ON r.request_id = q.id
  WHERE q.id = '<request_id>';"
```

- **Success:** a `COMPLETED` request has exactly one `inference_results` row.
- **Failure:** a `FAILED` request has none.
- **Expiry:** results become eligible for deletion 30 minutes after creation (`expires_at`),
  but nothing deletes them yet.

The scheduler also logs `input_tokens`, `generated_tokens`, the execution time and tokens per
second for every completed request. Time to first token is not measured, because generation
isn't streamed yet.

### Run the tests without downloading the model

The test suite always uses the simulated engine, even if `.env` sets
`INFERENCE_ENGINE=huggingface`. The Hugging Face engine is tested with fake tokenizer and model
objects, so no model is downloaded or loaded. Tests need the separate test database described
in [Testing](#testing).

### The resource manager is still a simulation

The scheduler's "GPU memory" and KV-cache blocks (see above) are **accounting only**. They don't
measure or limit the real CPU RAM used by PyTorch, and the KV-block sizing doesn't come from
Qwen's architecture. A request can be refused for lack of simulated KV blocks while plenty of
RAM is free, or the process can use more RAM than the simulation suggests.

## Testing

Automated tests run against their own PostgreSQL container, `test-db`, and never touch the
development database.

| | Development (`db`) | Tests (`test-db`) |
|---|---|---|
| Host port | 5432 | 15432 |
| Database | `inferenceflow` | `inferenceflow_test` |
| Volume | `inference_flow_postgres_data` | `inference_flow_test_postgres_data` |
| Configured by | `DATABASE_URL` | `TEST_DATABASE_URL` |

### Required configuration

Set `TEST_DATABASE_URL`, either in the shell (checked first) or in `backend/.env`:

```
TEST_DATABASE_URL=postgresql+psycopg://inferenceflow_test:test-only-password@localhost:15432/inferenceflow_test
```

Pytest stops before running any test if `TEST_DATABASE_URL`:

- is missing (it **never** falls back to `DATABASE_URL`),
- names a database that doesn't end in `_test`, or
- points at the same server (host and port) as `DATABASE_URL`, or
- reaches a database without the test identity marker (see below).

### Identity marker

Pytest also checks, **before migrating or truncating anything**, that the database is really the
`test-db` container. That database carries the comment `inferenceflow-test-db`, which
`docker/test-db/initdb/01-identity.sql` sets when the container first initializes its volume.
A database without that marker (another project's server, say) is refused. Pytest only reads
the marker; it never sets it.

A test volume created before this marker existed needs it applied once, inside the test
container. Never run this against the development database:

```bash
docker compose up -d --wait test-db    # recreates test-db with the init script mounted
docker exec inferenceflow-test-db psql -v ON_ERROR_STOP=1 -U inferenceflow_test -d inferenceflow_test \
    -f /docker-entrypoint-initdb.d/01-identity.sql
```

### Start the test database

```bash
docker compose up -d --wait test-db     # starts only test-db and waits for its health check
```

### Run the tests

```bash
cd backend
source env/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

At the start of every run, pytest checks that the test database is reachable and carries the
identity marker. It then applies
the Alembic migrations (`alembic upgrade head`) and empties the application tables. It never
empties `alembic_version`, and never creates or drops databases. Tests can run while a
development server is using the development database.

To check migrations against the test database:

```bash
DATABASE_URL="$TEST_DATABASE_URL" alembic check
```

### Stop the test database

```bash
docker compose stop test-db
```

### Reset the test database

Only needed if the test database gets into a bad state, for example after editing a migration
that was already applied. The recreated volume gets the identity marker automatically:

```bash
docker compose rm --stop --force test-db
docker volume rm inference_flow_test_postgres_data
docker compose up -d --wait test-db
```

> **Do not use `docker compose down -v`.** It removes *every* volume in this project, including
> `inference_flow_postgres_data`, and deletes the development database.
