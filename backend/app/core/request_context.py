"""Per-request identifiers, available to all code handling a request (and to its log records).

Every HTTP request gets two identifiers, set by RequestContextMiddleware before routing:

- `request_id` (external): the correlation ID shown to clients in the X-Request-ID response
  header. It is the client's own X-Request-ID when that is valid, otherwise generated. It is
  only ever used in headers and logs, never as a database key: clients reuse it (retries), and
  it is arbitrary text.
- `gateway_request_id` (internal): a UUID the gateway always generates itself. It is the key of
  the request's quota reservation and usage record, and the ID in `chatcmpl-<hex>`.

When the client sends no valid X-Request-ID, the external ID is the internal UUID's string
form, so a request has one visible identifier. The two are still separate values here.

They live in a ContextVar: each request runs in its own task with its own context, and tasks
and worker threads started while handling it (Starlette's streaming task, run_in_threadpool)
inherit a copy, so a streamed response keeps its IDs until it ends.
"""

import re
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


@dataclass(frozen=True)
class RequestIds:
    request_id: str
    gateway_request_id: uuid.UUID


_current: ContextVar[RequestIds | None] = ContextVar("inferenceflow_request_ids", default=None)


def valid_client_request_id(value: str | None) -> str | None:
    """The client's X-Request-ID if it is 1-64 characters of [A-Za-z0-9._:-], else None."""
    if value is not None and _VALID_REQUEST_ID.fullmatch(value):
        return value
    return None


def new_request_ids(client_request_id: str | None) -> RequestIds:
    gateway_request_id = uuid.uuid4()  # never derived from anything the client sent
    request_id = valid_client_request_id(client_request_id) or str(gateway_request_id)
    return RequestIds(request_id=request_id, gateway_request_id=gateway_request_id)


def bind(ids: RequestIds) -> Token:
    return _current.set(ids)


def unbind(token: Token) -> None:
    _current.reset(token)


def current() -> RequestIds | None:
    """The current request's IDs, or None outside a request (startup, CLI, direct calls)."""
    return _current.get()


def gateway_request_id() -> uuid.UUID:
    """The current request's internal UUID. Outside a request (the route called without the
    middleware, e.g. directly from a test) a fresh UUID, so a database key always exists."""
    ids = _current.get()
    return ids.gateway_request_id if ids is not None else uuid.uuid4()
