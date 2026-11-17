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

## API key authentication

The gateway endpoints require an API key:

| Endpoint | Authentication |
|---|---|
| `POST /v1/chat/completions` | **API key required** |
| `GET /v1/models` | **API key required** |
| `POST /generate` | **None.** Legacy, unprotected endpoint (see below) |

Send the key as a Bearer token:

```bash
curl localhost:8000/v1/models -H "Authorization: Bearer $INFERENCEFLOW_API_KEY"
```

A missing, malformed, unknown or revoked key gets `401` with `WWW-Authenticate: Bearer` and the
same `{"detail": "Missing or invalid API key"}`, so the response doesn't reveal whether a key
ever existed. Authentication runs before the request body is validated and before any provider
call.

### Keys and applications

Every key belongs to a **client application**. A key looks like `if_k3F9xQ2m_<43 random
characters>`:

- `if_k3F9xQ2m` is the **key prefix**. It isn't secret; it identifies the key in listings and
  revocation.
- The rest is the secret (256 bits).
- Only the SHA-256 hash of the whole key is stored. The plaintext is shown once, when the key is
  created, and can't be recovered afterwards.

### Managing keys

Run from `backend/`. The commands act on the database in `DATABASE_URL`:

```bash
python -m app.cli create-app billing-service
python -m app.cli create-key --app billing-service --name production   # prints the key once
python -m app.cli list-keys [--app billing-service]                     # never shows secrets
python -m app.cli revoke-key if_k3F9xQ2m                                # takes effect immediately
```

`create-key` refuses unknown applications, so create the application first. There are no HTTP
endpoints for managing keys.

### Applying the schema to the development database

The API key tables come from migration `a2916f33ef9c`. Neither the test suite nor any automated
step migrates the development database, so apply it yourself when you're ready. Until then,
`/v1/*` requests against the development server fail, because the tables don't exist.

```bash
cd backend
alembic current          # expect 681491804915
alembic upgrade head     # creates client_applications and api_keys
alembic current          # expect a2916f33ef9c (head)
```

The migration only adds two tables and changes no existing data. `alembic downgrade 681491804915`
removes them again, along with every key.

### Legacy `/generate` is not protected

`POST /generate` (the original queue and scheduler pipeline) does **not** require an API key and
is kept only for compatibility. Don't expose it to untrusted networks. Client applications should
use `/v1/chat/completions`.

## Rate limiting

Every `/v1` request is rate-limited **per API key**, after the key is authenticated:

| Endpoint | Authentication | Rate limit |
|---|---|---|
| `POST /v1/chat/completions` | API key | per key |
| `GET /v1/models` | API key | per key |
| any future `/v1` route | API key (inherited) | per key (inherited) |
| `POST /generate` | **none** | **none** (legacy, see above) |

### How it works

Each key has a **token bucket** in Redis under `inferenceflow:ratelimit:<api key ID>` (never
the key itself):

- The bucket holds up to `RATE_LIMIT_BURST` tokens and refills continuously at
  `RATE_LIMIT_REQUESTS_PER_MINUTE / 60` tokens per second.
- Each request takes one token. A key can burst up to the capacity, then sustain the
  per-minute rate.
- The whole check (read, refill, consume, write, set expiry) is **one Lua script**, which Redis
  runs atomically. Concurrent requests, from any number of workers, can't take more tokens than
  the bucket holds.
- The script uses Redis's own clock, so workers with different clocks agree.
- Idle buckets expire once they would be full again anyway.

Order of checks for a `/v1` request: authenticate the key (401), then check the rate limit (429
or 503), then validate the body (422), then call the provider. A rejected request never reaches
the provider, and an unauthenticated one never reaches Redis.

### Responses

Accepted requests include:

| Header | Meaning |
|---|---|
| `X-RateLimit-Limit` | bucket capacity (`RATE_LIMIT_BURST`) |
| `X-RateLimit-Remaining` | whole tokens left after this request |
| `X-RateLimit-Reset` | seconds until the bucket is full again |

When a key is out of tokens, the response is **`429 Too Many Requests`** with
`{"detail": "Rate limit exceeded"}`, the headers above, and `Retry-After` (seconds until the next
token, rounded up).

If Redis is unreachable or doesn't answer within `REDIS_TIMEOUT_SECONDS`, rate limiting
**fails closed**: the request gets **`503`** with `{"detail": "Rate limiting is temporarily
unavailable"}` and is not passed to the provider.

### Configuration

| Variable | Default | |
|---|---|---|
| `REDIS_URL` | `redis://localhost:16379/0` | the `redis` Compose service, from the host-run backend |
| `REDIS_TIMEOUT_SECONDS` | `0.5` | connect and command timeout |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | sustained rate per key |
| `RATE_LIMIT_BURST` | `10` | bucket capacity per key |

The backend runs on the host, so it reaches Redis on the published port: `redis://localhost:16379/0`.
If the backend is ever run **inside** Docker Compose, use the service name instead:
`redis://redis:6379/0`. The limits are global settings, enforced separately for each key.

### Start Redis

```bash
docker compose up -d --wait redis          # development Redis on localhost:16379
```

Rate-limit state is short-lived, so this Redis keeps nothing on disk. Restarting it resets every
bucket.

## Token quotas and usage accounting

Three separate controls apply to a `/v1/chat/completions` request:

| | What it limits | Where | Resets |
|---|---|---|---|
| **Rate limit** | requests per key per minute | Redis token bucket | refills continuously |
| **Token quota** | total tokens per **application** | PostgreSQL `token_quotas` | never automatically |
| **Actual usage** | nothing: it's what the provider reports it used | PostgreSQL `usage_records` | it's a record |

Every client application has one token quota, shared by all of its keys:

- `limit_tokens`: the allowance.
- `consumed_tokens`: tokens charged so far.
- `reserved_tokens`: tokens held by requests that are still running.
- `available = max(0, limit - consumed - reserved)`.

### How a request uses the quota

1. **Estimate.** The gateway can't tokenize for every model, so it estimates conservatively:
   the UTF-8 byte length of every message, plus 16 tokens per message and 64 per request for
   chat-template framing, plus `max_tokens` (or `MAX_GENERATION_TOKENS` if the request doesn't set
   it). Byte-level BPE tokenizers (GPT, Llama 3, Qwen) never produce more tokens than bytes, so
   for them this is an upper bound. It is **not** exact, and other tokenizers aren't guaranteed to
   stay under it.
2. **Reserve** (one transaction). The application's quota row is locked with `SELECT ... FOR
   UPDATE`. If the estimate doesn't fit in `available`, the request is rejected: `429` with code
   `insufficient_quota`, and the provider is never called. Otherwise a reservation is recorded,
   `reserved_tokens` grows, and the transaction commits. That releases the database connection
   **before** the provider is called.
3. **Call the provider.** No database transaction or connection is held while it runs.
4. **Settle** (one transaction). The reservation's tokens move from `reserved` to `consumed`,
   according to the table below, and a usage record is written.

| What happened | Charged | Usage record |
|---|---|---|
| Provider reported usage | the **actual** total, even if above the reservation | yes, with token counts |
| Success, but no usage reported | the full reservation (conservative) | yes; counts `null`, `usage_reported: false` |
| Provider failed and certainly did nothing (refused the request, connection refused) | nothing: reservation `released` | no |
| Provider failed and may have run (read timeout, dropped connection, 5xx, unreadable reply) | the full reservation, or the actual usage if it was reported | yes, `status: failed` |

The provider adapter decides whether a failure may have consumed tokens, because that's
provider knowledge. Token counts are never invented: unreported usage stays `null`, and the
conservative charge appears only in `charged_tokens`.

### The quota is a soft limit

The quota is a **soft quota**, not a guaranteed spending cap. It's checked **before** a request
runs, against an estimate, and then the gateway charges what the provider **actually** used,
which can be more.

Example:

| | Tokens |
|---|---|
| Available before the request | 5,000 |
| Reserved (the estimate) | 5,000 |
| Actually used, as reported by the provider | 6,500 |
| Charged | **6,500** |

After this request, `consumed_tokens` is 1,500 above the limit. `available_tokens` shows **0**
(never negative), and every further request gets `insufficient_quota` until the limit is raised.

This is deliberate. The provider has already done, and billed, the work, so recording less than
was really used would only hide real consumption. Accurate accounting matters more here than a
hard cap. How far over the limit an application can go:

- per request, by however much its actual usage exceeds its estimate;
- with several requests running at once, by the sum of those overruns.

The estimate includes the request's full output allowance (`max_tokens`, or
`MAX_GENERATION_TOKENS`), so the prompt estimate is the main way to go over. Any overrun is
visible: compare `charged_tokens` with the reservation, or `consumed_tokens` with
`limit_tokens` in `GET /v1/usage/quota`.

A guaranteed hard cap would need the provider to stop generating at the remaining budget. That
isn't implemented.

**Exactly once.** Settlement locks the quota row, then the reservation, and does nothing if the
reservation was already settled. Each request has a unique `request_id`, which is also the
`chatcmpl-...` ID returned to the client. Both reservations and usage records are unique per
request, so settling the same request again can never charge it twice. The counters only change
together with the rows they summarize, inside that one locked transaction.

### Failures and recovery

- **Database down when reserving:** `503 {"detail": "Quota accounting is temporarily
  unavailable"}`; the provider is not called.
- **Database down when settling, or a process crash:** the response is still returned when
  possible, since the provider already did the work. The reservation stays `reserved` and keeps
  holding its tokens. The same applies if a request is cancelled mid-call. (Whether a client
  disconnect cancels a non-streamed request depends on the server; this hasn't been verified
  here.) Recover it with
  ```bash
  python -m app.cli expire-reservations --older-than-minutes 60
  ```
  which settles every reservation older than that as `expired`, charging its reserved tokens
  (the actual usage is unknown). Run it periodically, for example from cron, with an age well
  above the provider timeout (`OPENAI_COMPATIBLE_TIMEOUT_SECONDS`), so it never settles a
  request that is still running. There is no background worker.

### Managing quotas

```bash
python -m app.cli create-app billing-service --token-limit 500000   # default: DEFAULT_TOKEN_QUOTA
python -m app.cli set-quota --app billing-service --limit 2000000
python -m app.cli show-quota --app billing-service
```

Quotas don't reset on their own. To give an application more tokens, raise its limit.

If an application's quota row has been deleted (only possible by hand), `set-quota` recreates it
with its counters **rebuilt from the reservation history**:
`reserved_tokens` = the tokens of reservations still `reserved`, and `consumed_tokens` = the
tokens charged by every settled one. Requests that were in flight then settle normally, and
`consumed` reflects past usage again. While the row is missing, chat requests get
`insufficient_quota`, and settling a reservation leaves it `reserved` until the row is back.

### `GET /v1/usage`

Usage of the **authenticated key's application** only (all of its keys). Query parameters:

| Parameter | Meaning |
|---|---|
| `start`, `end` | ISO 8601 times; `start` is inclusive, `end` is exclusive; UTC if no offset is given |
| `model`, `provider` | exact match |
| `limit`, `offset` | page of records, newest first (`limit` 1–200, default 50) |

```json
{
  "object": "usage",
  "filters": {"start": null, "end": null, "model": null, "provider": null},
  "summary": {
    "requests": 2,
    "requests_without_reported_usage": 0,
    "prompt_tokens": 24,
    "completion_tokens": 61,
    "total_tokens": 85,
    "charged_tokens": 85
  },
  "data": [
    {
      "request_id": "chatcmpl-4f2c...",
      "created_at": "2026-11-16T18:20:33.511Z",
      "key_prefix": "if_k3F9xQ2m",
      "provider": "openai_compatible",
      "model": "llama-3.1-8b-instant",
      "status": "succeeded",
      "usage_reported": true,
      "prompt_tokens": 12,
      "completion_tokens": 30,
      "total_tokens": 42,
      "charged_tokens": 42,
      "latency_ms": 812.4
    }
  ],
  "pagination": {"limit": 50, "offset": 0, "has_more": true}
}
```

The summary covers every matching record, not just the page. The token sums include
provider-reported counts only; `charged_tokens` is what came off the quota.

### `GET /v1/usage/quota`

```json
{"object": "quota", "limit_tokens": 1000000, "consumed_tokens": 85, "reserved_tokens": 0, "available_tokens": 999915}
```

An application without a quota row gets `404` here, and `insufficient_quota` on chat requests.
Both usage endpoints are on the `/v1` router, so they require an API key and are rate-limited.
`GET /v1/models` doesn't use quota.

### When quota runs out

```json
{"detail": {"code": "insufficient_quota", "message": "Not enough token quota left for this request",
            "required_tokens": 92, "available_tokens": 40}}
```

It's a `429` like the rate limit, but it has **no** `Retry-After`: the quota doesn't refill on
its own.

### Applying the schema

Migration `77dc84617c96` adds `token_quotas`, `quota_reservations` and `usage_records`, and gives
every existing application a quota of 1,000,000 tokens. Apply it to the development database
deliberately:

```bash
cd backend && alembic upgrade head      # expect 77dc84617c96 (head) afterwards
```

### Limitations

- **Soft quota.** Usage above an estimate is charged in full, so consumption can go past the limit
  (see "The quota is a soft limit" above).
- **Estimates aren't exact.** A request can be rejected even though slightly less than its
  estimate would really have been needed.
- **Missing usage costs the full reservation.** If a provider never reports usage, every request
  costs its full reservation.
- **Stale reservations need the recovery command.** Until `expire-reservations` runs, they keep
  holding tokens.
- **No resets, billing or per-key quotas.** Quotas are per application.
- **No streaming yet.** Quota handling covers non-streamed requests only, because streaming isn't
  implemented.

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

### Test Redis

Tests use their own Redis container, `test-redis` (host port **16380**), never the development
`redis` (16379). Set `TEST_REDIS_URL` in the shell or `backend/.env`:

```
TEST_REDIS_URL=redis://localhost:16380/0
```

Pytest stops before any test runs if `TEST_REDIS_URL` is missing (it **never** falls back to
`REDIS_URL`), or points at the same Redis server (host and port) as `REDIS_URL`. A different
database number on the same server isn't accepted.

Before touching anything, pytest also checks that the server carries the key
`inferenceflow:test-redis:identity` = `inferenceflow-test-redis`.
`docker/test-redis/start.sh` sets it each time the container starts, and pytest only reads it.
Pytest then deletes only rate-limit buckets (`inferenceflow:ratelimit:*`), at the start and end
of the run. It never flushes the server.

```bash
docker compose up -d --wait test-redis     # start
docker compose stop test-redis             # stop
```

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
docker compose up -d --wait test-db test-redis   # the test database and test Redis
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
