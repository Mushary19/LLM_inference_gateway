"""RequestContextMiddleware: request IDs, the X-Request-ID response header and one
`http_request_finished` log line per HTTP request.

A pure ASGI middleware: `receive` is passed through untouched and `send` is only wrapped to
add the header and note the status, so streaming responses and their disconnect detection
work exactly as without it (Starlette's BaseHTTPMiddleware would run the app in a separate
task and change both).

It wraps the whole Starlette middleware stack, outside ServerErrorMiddleware (see
InferenceFlowApp in app/main.py). That is what makes the 500 response ServerErrorMiddleware
builds for an unhandled exception pass through this `send` and carry X-Request-ID, without
this middleware building any response of its own or catching anything: exceptions are
re-raised unchanged, so the server still logs and handles them as before.
"""

import logging
import time
from typing import Any

import anyio
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core import request_context
from app.core.request_context import REQUEST_ID_HEADER
from app.core.structured_logging import log_event

logger = logging.getLogger(__name__)

_HEADER_NAME = REQUEST_ID_HEADER.lower().encode("latin-1")


def _client_request_id(scope: Scope) -> str | None:
    """The first X-Request-ID header's value, if any (validated by request_context)."""
    for name, value in scope.get("headers", ()):
        if name.lower() == _HEADER_NAME:
            return value.decode("latin-1")
    return None


def _route_template(scope: Scope) -> str | None:
    """The matched route's path template (e.g. /v1/chat/completions), never the raw path.
    None when no route matched."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else None


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan (and websockets, which the app doesn't serve)
            await self.app(scope, receive, send)
            return

        ids = request_context.new_request_ids(_client_request_id(scope))
        header_value = ids.request_id.encode("latin-1")  # validated or generated: always ASCII
        status_code: int | None = None
        started = time.monotonic()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = [(k, v) for k, v in message.get("headers", ()) if k.lower() != _HEADER_NAME]
                headers.append((_HEADER_NAME, header_value))
                message = {**message, "headers": headers}
            await send(message)

        token = request_context.bind(ids)
        try:
            try:
                await self.app(scope, receive, send_with_request_id)
            except BaseException as exc:
                self._log_finished(scope, status_code, started, exc)
                raise
            self._log_finished(scope, status_code, started, None)
        finally:
            request_context.unbind(token)

    @staticmethod
    def _log_finished(scope: Scope, status_code: int | None, started: float, exc: BaseException | None) -> None:
        fields: dict[str, Any] = {
            "method": scope.get("method"),
            "route": _route_template(scope),
            "status_code": status_code,
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
        }
        level = logging.INFO
        exc_info = None
        if exc is not None:
            fields["error_type"] = type(exc).__name__
            if isinstance(exc, anyio.get_cancelled_exc_class()) or isinstance(exc, (ClientDisconnect, OSError)):
                level = logging.INFO  # cancellation or a client that went away: not a gateway failure
            else:
                # An exception no handler turned into a response; the server logs it too, but
                # without the request IDs.
                level = logging.ERROR
                exc_info = (type(exc), exc, exc.__traceback__)
        elif status_code is not None and status_code >= 500:
            level = logging.WARNING
        message = f"{fields['method']} {fields['route'] or '(no route)'} finished with status {status_code}"
        log_event(logger, level, "http_request_finished", message, exc_info=exc_info, **fields)
