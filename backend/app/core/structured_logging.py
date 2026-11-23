"""Logging for the application's own loggers (everything under "app").

Without this, nothing configures the "app" loggers: under uvicorn's logging configuration
(which only sets up uvicorn's own loggers) their INFO records are dropped and WARNING and
above reach Python's last-resort handler as bare messages.

configure_logging() gives the "app" logger one stdout handler, a level (LOG_LEVEL) and a
format (LOG_FORMAT): one JSON object per line by default, or readable text for local
development. uvicorn's loggers are left as uvicorn configures them. Records still propagate
to the root logger, which has no handlers under uvicorn, so nothing is printed twice; a
custom uvicorn --log-config that gives the root logger a handler would print app records
twice and should exclude them.

Every record gets the current request's IDs (see app/core/request_context.py). Structured
events are logged with log_event(): an event name plus fields, which the formatters write
as top-level keys. Never pass secrets, prompts, completions or raw bodies as fields.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Literal

from app.core import request_context

LogFormat = Literal["json", "text"]

APP_LOGGER = "app"
_EVENT = "event"
_FIELDS = "fields"


def log_event(
    logger: logging.Logger, level: int, event: str, message: str, /, *, exc_info: Any = None, **fields: Any
) -> None:
    """Log a structured event: `event` is a stable name, `fields` become top-level keys (a
    field named like a standard key, e.g. "message", never replaces it)."""
    logger.log(level, message, exc_info=exc_info, extra={_EVENT: event, _FIELDS: fields})


class RequestContextFilter(logging.Filter):
    """Adds request_id and gateway_request_id (None outside a request) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        ids = request_context.current()
        record.request_id = ids.request_id if ids is not None else None
        record.gateway_request_id = str(ids.gateway_request_id) if ids is not None else None
        return True


def _timestamp(record: logging.LogRecord) -> str:
    moment = datetime.fromtimestamp(record.created, tz=timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _record_fields(record: logging.LogRecord) -> dict[str, Any]:
    fields = getattr(record, _FIELDS, None)
    return dict(fields) if isinstance(fields, dict) else {}


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Keys: timestamp, level, logger, message, then event,
    request_id and gateway_request_id when present, the event's fields, and exception
    (the formatted traceback) when the record has one."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (_EVENT, "request_id", "gateway_request_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        for key, value in _record_fields(record).items():
            payload.setdefault(key, value)  # a field never overwrites the keys above
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """`<timestamp> <LEVEL> <logger> [request_id=...] <message> key=value ...` and the traceback."""

    def format(self, record: logging.LogRecord) -> str:
        parts = [_timestamp(record), record.levelname, record.name]
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            parts.append(f"[request_id={request_id} gateway_request_id={getattr(record, 'gateway_request_id', None)}]")
        parts.append(record.getMessage())
        event = getattr(record, _EVENT, None)
        if event is not None:
            parts.append(f"event={event}")
        parts.extend(f"{key}={json.dumps(value, default=str)}" for key, value in _record_fields(record).items())
        text = " ".join(parts)
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            text += "\n" + self.formatStack(record.stack_info)
        return text


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is at the time (like logging.lastResort does for
    stderr), so replacing sys.stdout (e.g. output capture in tests) never leaves it
    writing to a closed stream."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        pass


def configure_logging(level: str = "INFO", log_format: LogFormat = "json") -> logging.Handler:
    """Configure the "app" logger. Idempotent: a previous handler installed here is replaced."""
    logger = logging.getLogger(APP_LOGGER)
    for handler in list(logger.handlers):
        if isinstance(handler, _StdoutHandler):
            logger.removeHandler(handler)
    handler = _StdoutHandler()
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    return handler
