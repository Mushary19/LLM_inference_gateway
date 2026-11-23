"""Incremental Server-Sent Events decoder for upstream provider streams.

Follows the WHATWG event-stream parsing rules, for the parts a chat completion stream uses:
- lines end with CRLF, LF or a bare CR (a CRLF split across two reads is one line ending);
- a UTF-8 BOM at the very start is skipped; invalid UTF-8 is replaced, not fatal;
- a blank line dispatches the event; lines starting with ":" are comments;
- "field:value" (one leading space of the value is dropped) or a bare "field" (empty value);
- every "data" field is appended to the event's data, the values joined with "\\n";
- "event", "id", "retry" and unknown fields are ignored;
- an event with no data field is not dispatched; an unfinished event at EOF is discarded.

One event (all its lines, including the line being read) may not exceed `max_event_bytes`,
so a misbehaving upstream can't make the gateway buffer without limit.
"""

import re

MAX_EVENT_BYTES = 1024 * 1024  # 1 MiB

_LINE_END = re.compile(rb"[\r\n]")
_BOM = "﻿"


class SSEEventTooLargeError(ValueError):
    """An upstream event exceeded the size limit."""


class SSEDecoder:
    def __init__(self, max_event_bytes: int = MAX_EVENT_BYTES) -> None:
        self._max_event_bytes = max_event_bytes
        self._line = bytearray()
        self._data: list[str] = []
        self._has_data = False
        self._event_bytes = 0
        self._skip_lf = False  # the previous read ended with CR: a leading LF belongs to it
        self._at_start = True

    def feed(self, chunk: bytes) -> list[str]:
        """Consume raw bytes; return the data of every event they complete, in order."""
        events: list[str] = []
        position = 0
        if chunk and self._skip_lf:
            self._skip_lf = False
            if chunk[0] == 0x0A:
                position = 1
        while position < len(chunk):
            match = _LINE_END.search(chunk, position)
            if match is None:
                self._line += chunk[position:]
                self._check_size()
                break
            end = match.start()
            self._line += chunk[position:end]
            self._check_size()
            self._end_line(events)
            position = end + 1
            if chunk[end] == 0x0D:
                if position < len(chunk):
                    if chunk[position] == 0x0A:
                        position += 1
                else:
                    self._skip_lf = True
        return events

    def close(self) -> None:
        """End of stream: an event without its terminating blank line is discarded."""
        self._line.clear()
        self._reset_event()

    def _check_size(self) -> None:
        if self._event_bytes + len(self._line) > self._max_event_bytes:
            raise SSEEventTooLargeError(f"upstream event larger than {self._max_event_bytes} bytes")

    def _reset_event(self) -> None:
        self._data.clear()
        self._has_data = False
        self._event_bytes = 0

    def _end_line(self, events: list[str]) -> None:
        raw = bytes(self._line)
        self._line.clear()
        line = raw.decode("utf-8", errors="replace")
        if self._at_start:
            self._at_start = False
            if line.startswith(_BOM):
                line = line[1:]
        if not line:
            if self._has_data:
                events.append("\n".join(self._data))
            self._reset_event()
            return
        self._event_bytes += len(raw) + 1
        if line.startswith(":"):
            return  # comment
        field, colon, value = line.partition(":")
        if colon and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
            self._has_data = True
        # "event", "id", "retry" and unknown fields: ignored.
