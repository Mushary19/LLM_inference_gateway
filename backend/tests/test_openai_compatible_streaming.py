"""OpenAICompatibleProvider.stream() and the SSE decoder, against mocked HTTP and loopback.

No real provider is ever contacted: upstreams are httpx MockTransports, or a local asyncio
server on 127.0.0.1 for the tests that need a real socket.
"""

import asyncio
import json
import logging
import time

import httpx2 as httpx
import pytest

from app.providers.base import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    InvalidProviderRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.providers.factory import create_provider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.registry import OpenAICompatibleProviderConfig, RegistryConfigError, parse_registry_config
from app.providers.sse import MAX_EVENT_BYTES, SSEDecoder, SSEEventTooLargeError

BASE_URL = "http://internal-llm.corp.example:8000/v1"
MODEL = "upstream-model"
SSE = {"content-type": "text/event-stream; charset=utf-8"}
ACCEPTED = ChatChunk(delta="")


def chat_request(**overrides) -> ChatRequest:
    return ChatRequest(**{"model": MODEL, "messages": (ChatMessage("user", "Hi"),), **overrides})


def event(payload) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n".encode()


def delta(content=None, *, role=None, finish=None, index=0, **fields) -> dict:
    d = {k: v for k, v in (("role", role), ("content", content)) if v is not None}
    return {"id": "chatcmpl-up", "object": "chat.completion.chunk", "model": MODEL,
            "choices": [{"index": index, "delta": {**d, **fields}, "finish_reason": finish}]}


def usage_event(prompt=5, completion=2) -> dict:
    return {"id": "chatcmpl-up", "choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                                         "total_tokens": prompt + completion}}


def normal_stream(usage=True) -> list[bytes]:
    parts = [event(delta("", role="assistant")), event(delta("Hel")), event(delta("lo")), event(delta(finish="stop"))]
    if usage:
        parts.append(event(usage_event()))
    return parts + [event("[DONE]")]


class Body(httpx.AsyncByteStream):
    """A streamed response body: yields parts (optionally slowly), then raises or hangs.
    Records whether the HTTP client closed it."""

    def __init__(self, parts, *, delay=0.0, then_raise: Exception | None = None, hang=False):
        self.parts = list(parts)
        self.delay = delay
        self.then_raise = then_raise
        self.hang = hang
        self.closed = False
        self.sent = 0

    async def __aiter__(self):
        for part in self.parts:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.sent += 1
            yield part
        if self.then_raise is not None:
            raise self.then_raise
        if self.hang:
            await asyncio.sleep(3600)

    async def aclose(self):
        self.closed = True


class Upstream:
    """Mock upstream for streamed requests; records requests."""

    def __init__(self, body: Body | None = None, *, status=200, headers=SSE, content=None, raises=None, wait=0.0):
        self.body = body if body is not None else Body(normal_stream())
        self.status = status
        self.headers = headers
        self.content = content
        self.raises = raises
        self.wait = wait
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.wait:
            await asyncio.sleep(self.wait)
        if self.raises is not None:
            raise self.raises
        if self.content is not None:
            return httpx.Response(self.status, headers=self.headers, content=self.content)
        return httpx.Response(self.status, headers=self.headers, stream=self.body)

    @property
    def payload(self) -> dict:
        return json.loads(self.requests[-1].content)


def make_provider(upstream, *, timeout_seconds=60.0, stream_usage=False) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        BASE_URL, api_key="sk-secret", timeout_seconds=timeout_seconds, transport=httpx.MockTransport(upstream),
        name="mock", stream_usage=stream_usage,
    )


async def _collect(provider, request, into):
    try:
        async for chunk in provider.stream(request):
            into.append(chunk)
    finally:
        await provider.aclose()


def collect(upstream, request=None, **provider_options) -> list[ChatChunk]:
    chunks: list[ChatChunk] = []
    asyncio.run(_collect(make_provider(upstream, **provider_options), request or chat_request(), chunks))
    return chunks


def collect_error(upstream, error_type=ProviderError, **provider_options):
    """Run the stream expecting a failure; returns (chunks before it, the exception)."""
    chunks: list[ChatChunk] = []
    with pytest.raises(error_type) as excinfo:
        asyncio.run(_collect(make_provider(upstream, **provider_options), chat_request(), chunks))
    return chunks, excinfo.value


# --- Normal streams ---


def test_normal_stream():
    upstream = Upstream(Body(normal_stream()))

    chunks = collect(upstream, chat_request(max_tokens=20, temperature=0.5))

    # Accepted first, then content (the upstream's role-only chunk carries nothing), then the
    # terminal chunk with finish_reason and usage.
    assert chunks == [ACCEPTED, ChatChunk("Hel"), ChatChunk("lo"), ChatChunk("", "stop", Usage(5, 2))]
    request = upstream.requests[0]
    assert request.method == "POST" and request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-secret"
    assert upstream.payload == {
        "model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": True,
        "max_tokens": 20, "temperature": 0.5,
    }
    assert upstream.body.closed


def test_stream_options_are_only_sent_when_stream_usage_is_enabled():
    default, enabled = Upstream(), Upstream()

    collect(default)
    collect(enabled, stream_usage=True)

    assert "stream_options" not in default.payload
    assert enabled.payload["stream_options"] == {"include_usage": True}


def test_rejected_usage_option_is_not_retried():
    upstream = Upstream(status=400, headers={"content-type": "application/json"},
                        content=json.dumps({"error": {"message": "unknown field stream_options"}}).encode())

    collect_error(upstream, InvalidProviderRequestError, stream_usage=True)

    assert len(upstream.requests) == 1


@pytest.mark.parametrize("upstream_reason, normalized", [("stop", "stop"), ("length", "length"), ("tool_calls", "stop")])
def test_finish_reasons_are_normalized(upstream_reason, normalized):
    chunks = collect(Upstream(Body([event(delta("x")), event(delta(finish=upstream_reason)), event("[DONE]")])))

    assert chunks[-1].finish_reason == normalized


def test_content_in_the_finish_chunk_is_kept():
    chunks = collect(Upstream(Body([event(delta("last", finish="stop")), event("[DONE]")])))

    assert chunks == [ACCEPTED, ChatChunk("last"), ChatChunk("", "stop", None)]


def test_groq_usage_in_x_groq_is_used():
    finish = {**delta(finish="stop"), "x_groq": {"id": "req", "usage": {"prompt_tokens": 9, "completion_tokens": 4}}}

    chunks = collect(Upstream(Body([event(delta("x")), event(finish), event("[DONE]")])))

    assert chunks[-1].usage == Usage(9, 4)


def test_top_level_usage_takes_precedence_over_x_groq():
    finish = {**delta(finish="stop"), "x_groq": {"usage": {"prompt_tokens": 9, "completion_tokens": 4}}}

    chunks = collect(Upstream(Body([event(finish), event(usage_event(1, 1)), event("[DONE]")])))

    assert chunks[-1].usage == Usage(1, 1)


def test_missing_usage_is_unreported(caplog):
    chunks = collect(Upstream(Body(normal_stream(usage=False))))

    assert chunks[-1] == ChatChunk("", "stop", None)
    assert "reported no usage" in caplog.text


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1, "completion_tokens": 2},
        {"prompt_tokens": "5", "completion_tokens": 2},
        {"prompt_tokens": True, "completion_tokens": 2},
        {"completion_tokens": 2},
        "lots",
        [1, 2],
    ],
)
def test_invalid_usage_is_unreported_and_does_not_break_the_stream(usage, caplog):
    body = Body([event(delta("ok")), event(delta(finish="stop")), event({"choices": [], "usage": usage}), event("[DONE]")])

    chunks = collect(Upstream(body))

    assert chunks == [ACCEPTED, ChatChunk("ok"), ChatChunk("", "stop", None)]
    assert "invalid usage" in caplog.text


def test_invalid_groq_usage_is_unreported():
    finish = {**delta(finish="stop"), "x_groq": {"usage": {"prompt_tokens": "x"}}}

    assert collect(Upstream(Body([event(finish), event("[DONE]")])))[-1].usage is None


def test_reasoning_fields_and_other_choices_are_dropped():
    parts = [
        event(delta(reasoning="thinking...")),
        event(delta(reasoning_content="more thinking")),
        event(delta("")),
        event(delta("other choice", index=1)),
        event(delta("answer")),
        event(delta(finish="stop")),
        event("[DONE]"),
    ]

    assert collect(Upstream(Body(parts))) == [ACCEPTED, ChatChunk("answer"), ChatChunk("", "stop", None)]


def test_events_after_done_are_not_read():
    body = Body(normal_stream() + [event("this is not JSON")])

    assert collect(Upstream(body))[-1] == ChatChunk("", "stop", Usage(5, 2))
    assert body.closed


def test_eof_after_finish_reason_without_done_is_complete():
    body = Body([event(delta("ok")), event(delta(finish="length")), event(usage_event())])

    assert collect(Upstream(body))[-1] == ChatChunk("", "length", Usage(5, 2))


# --- SSE framing ---


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["LF", "CRLF", "CR"])
def test_line_endings(newline):
    text = "".join(f"data: {json.dumps(p)}{newline}{newline}" for p in (delta("a"), delta("b"), delta(finish="stop")))
    text += f"data: [DONE]{newline}{newline}"

    chunks = collect(Upstream(Body([text.encode()])))

    assert chunks == [ACCEPTED, ChatChunk("a"), ChatChunk("b"), ChatChunk("", "stop", None)]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["LF", "CRLF", "CR"])
def test_every_read_boundary(newline):
    """Split TCP reads: deliver the stream one byte at a time (CRLF split between reads too)."""
    text = "".join(f"data: {json.dumps(p)}{newline}{newline}" for p in (delta("é"), delta(finish="stop")))
    raw = text.encode()

    chunks = collect(Upstream(Body([raw[i:i + 1] for i in range(len(raw))])))

    assert chunks == [ACCEPTED, ChatChunk("é"), ChatChunk("", "stop", None)]


def test_multiple_data_lines_are_joined_with_newlines():
    # Valid JSON only once the lines are joined with "\n".
    first = 'data: {"choices": [{"index": 0,\ndata:  "delta": {"content": "multi"},\ndata: "finish_reason": null}]}\n\n'

    chunks = collect(Upstream(Body([first.encode(), event(delta(finish="stop"))])))

    assert chunks[1] == ChatChunk("multi")


def test_comments_and_unsupported_fields_are_ignored():
    text = (
        ": keep-alive\n"
        "event: message\n"
        "id: 7\n"
        "retry: 1000\n"
        "unknown-field: whatever\n"
        "data\n"  # a field without a colon: "data" with an empty value
        f"data:{json.dumps(delta('x'))}\n"  # no space after the colon
        "\n"
        ":\n"
        "event: ping\n"
        "\n"  # an event without data is not dispatched
    )

    chunks = collect(Upstream(Body([text.encode(), event(delta(finish="stop"))])))

    assert chunks == [ACCEPTED, ChatChunk("x"), ChatChunk("", "stop", None)]


def test_a_leading_byte_order_mark_is_ignored():
    chunks = collect(Upstream(Body([b"\xef\xbb\xbf" + event(delta("x")), event(delta(finish="stop"))])))

    assert chunks[1] == ChatChunk("x")


def test_unfinished_event_at_eof_is_discarded():
    # The finish event never gets its terminating blank line, so it is not dispatched.
    body = Body([event(delta("x")), f"data: {json.dumps(delta(finish='stop'))}\n".encode()])

    chunks, error = collect_error(Upstream(body))

    assert chunks == [ACCEPTED, ChatChunk("x")]
    assert "incomplete" in str(error) and error.may_have_consumed_tokens


def test_eof_before_finish_reason_is_incomplete():
    chunks, error = collect_error(Upstream(Body([event(delta("partial"))])))

    assert chunks == [ACCEPTED, ChatChunk("partial")]
    assert type(error) is ProviderError and error.may_have_consumed_tokens


def test_done_before_finish_reason_is_incomplete():
    _, error = collect_error(Upstream(Body([event(delta("partial")), event("[DONE]")])))

    assert "incomplete" in str(error)


@pytest.mark.parametrize("data", ["{not json", "[1, 2]", '"text"', '{"choices": "nope"}'])
def test_malformed_events_fail_the_stream(data):
    chunks, error = collect_error(Upstream(Body([event(delta("x")), event(data)])))

    assert chunks == [ACCEPTED, ChatChunk("x")]
    assert type(error) is ProviderError and error.may_have_consumed_tokens


def test_upstream_error_event_fails_the_stream():
    _, error = collect_error(Upstream(Body([event(delta("x")), event({"error": {"message": "overloaded"}})])))

    assert type(error) is ProviderError and "overloaded" in str(error)


def test_event_size_limit():
    big = "y" * (MAX_EVENT_BYTES - 200)
    ok = event(delta(big))

    assert collect(Upstream(Body([ok[i:i + 65536] for i in range(0, len(ok), 65536)] + [event(delta(finish="stop"))])))[1].delta == big

    too_big = event(delta("z" * MAX_EVENT_BYTES))
    _, error = collect_error(Upstream(Body([too_big])))
    assert "larger than" in str(error)


def test_an_endless_line_is_rejected_without_buffering_it_all():
    body = Body([b"data: " + b"x" * 65536] * 100)  # 6.4 MB, no newline ever

    _, error = collect_error(Upstream(body))

    assert "larger than" in str(error)
    assert body.sent <= MAX_EVENT_BYTES // 65536 + 2  # stopped reading at the limit


# --- Errors before the stream is accepted ---


@pytest.mark.parametrize(
    "status, error_type, consumed",
    [
        (400, InvalidProviderRequestError, False),
        (401, ProviderError, False),
        (404, ModelNotFoundError, False),
        (429, ProviderUnavailableError, False),
        (500, ProviderError, True),
        (503, ProviderUnavailableError, True),
        (504, ProviderTimeoutError, True),
    ],
)
def test_http_errors_are_raised_before_the_accepted_chunk(status, error_type, consumed):
    body = Body([json.dumps({"error": {"message": f"upstream says {status}"}}).encode()])
    upstream = Upstream(body, status=status, headers={"content-type": "application/json"})

    chunks, error = collect_error(upstream, error_type)

    assert chunks == []  # nothing, not even the accepted chunk
    assert type(error) is error_type
    assert error.status_code == status and f"upstream says {status}" in str(error)
    assert error.may_have_consumed_tokens is consumed
    assert body.closed


def test_large_error_bodies_are_only_partly_read():
    body = Body([b"x" * 65536] * 50)

    collect_error(Upstream(body, status=500, headers={"content-type": "text/plain"}))

    assert body.sent <= 2 and body.closed


@pytest.mark.parametrize("content_type", ["application/json", "text/plain", ""])
def test_non_event_stream_responses_are_rejected(content_type):
    upstream = Upstream(status=200, headers={"content-type": content_type} if content_type else {},
                        content=json.dumps({"choices": []}).encode())

    chunks, error = collect_error(upstream)

    assert chunks == []
    assert type(error) is ProviderError and error.may_have_consumed_tokens


@pytest.mark.parametrize(
    "raised, error_type, consumed",
    [
        (httpx.ConnectError("refused"), ProviderUnavailableError, False),
        (httpx.ConnectTimeout("slow connect"), ProviderTimeoutError, False),
        (httpx.PoolTimeout("pool full"), ProviderTimeoutError, False),
        (httpx.ReadTimeout("no headers"), ProviderTimeoutError, True),
        (httpx.RemoteProtocolError("bad response"), ProviderUnavailableError, True),
    ],
)
def test_connection_failures(raised, error_type, consumed):
    chunks, error = collect_error(Upstream(raises=raised), error_type)

    assert chunks == []
    assert type(error) is error_type and error.may_have_consumed_tokens is consumed


# --- Failures after the stream was accepted ---


@pytest.mark.parametrize(
    "raised, error_type",
    [
        (httpx.ReadError("reset"), ProviderUnavailableError),
        (httpx.RemoteProtocolError("peer closed"), ProviderUnavailableError),
        (httpx.ReadTimeout("stalled"), ProviderTimeoutError),
    ],
)
def test_transport_failures_mid_stream(raised, error_type):
    body = Body([event(delta("partial"))], then_raise=raised)

    chunks, error = collect_error(Upstream(body), error_type)

    assert chunks == [ACCEPTED, ChatChunk("partial")]
    assert type(error) is error_type and error.may_have_consumed_tokens
    assert body.closed


def test_transport_failure_after_the_finish_reason_still_completes(caplog):
    body = Body([event(delta("x")), event(delta(finish="stop"))], then_raise=httpx.ReadError("reset"))

    chunks = collect(Upstream(body))

    assert chunks[-1] == ChatChunk("", "stop", None)
    assert "did not close after finishing" in caplog.text


# --- The total deadline (timeout_seconds) ---


def test_the_deadline_covers_the_whole_stream():
    """A trickling stream never trips a per-read timeout; the total deadline still ends it."""
    body = Body([event(delta("."))] * 1000, delay=0.05)

    started = time.monotonic()
    chunks, error = collect_error(Upstream(body), ProviderTimeoutError, timeout_seconds=0.5)
    elapsed = time.monotonic() - started

    assert 0.4 < elapsed < 2
    assert 3 <= len(chunks) < 20 and error.may_have_consumed_tokens
    assert body.closed


def test_the_deadline_covers_waiting_for_the_response():
    chunks, error = collect_error(Upstream(wait=5), ProviderTimeoutError, timeout_seconds=0.3)

    assert chunks == [] and error.may_have_consumed_tokens


def test_a_hang_after_the_finish_reason_ends_complete_at_the_deadline():
    body = Body([event(delta("x")), event(delta(finish="stop"))], hang=True)

    chunks = collect(Upstream(body), timeout_seconds=0.3)

    assert chunks[-1] == ChatChunk("", "stop", None) and body.closed


# --- Closing and cancellation ---


def test_closing_after_the_accepted_chunk_closes_the_response():
    body = Body(normal_stream())
    provider = make_provider(Upstream(body))

    async def scenario():
        stream = provider.stream(chat_request())
        assert await stream.__anext__() == ACCEPTED
        await stream.aclose()
        await provider.aclose()

    asyncio.run(scenario())
    assert body.closed and body.sent == 0


def test_closing_mid_stream_closes_the_response():
    body = Body(normal_stream(), hang=True)
    provider = make_provider(Upstream(body))

    async def scenario():
        stream = provider.stream(chat_request())
        assert [await stream.__anext__() for _ in range(2)] == [ACCEPTED, ChatChunk("Hel")]
        await stream.aclose()
        await provider.aclose()

    asyncio.run(scenario())
    assert body.closed


def test_cancellation_while_waiting_for_upstream_closes_the_response():
    body = Body([event(delta("x"))], hang=True)
    provider = make_provider(Upstream(body))

    async def scenario():
        stream = provider.stream(chat_request())
        await stream.__anext__()
        await stream.__anext__()
        task = asyncio.create_task(stream.__anext__())  # now waiting on the hanging upstream
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await provider.aclose()

    asyncio.run(scenario())
    assert body.closed


def test_the_stream_can_be_continued_and_closed_in_other_tasks():
    """The gateway starts the stream in the route's task, continues it in Starlette's
    streaming task and may close it from another: no cancel scope may span those."""
    body = Body(normal_stream(), delay=0.01)
    provider = make_provider(Upstream(body), timeout_seconds=30)

    async def scenario():
        stream = provider.stream(chat_request())
        first = await asyncio.create_task(stream.__anext__())
        second = await asyncio.create_task(stream.__anext__())
        third = await asyncio.create_task(stream.__anext__())
        await asyncio.create_task(stream.aclose())
        await provider.aclose()
        return [first, second, third]

    assert asyncio.run(scenario()) == [ACCEPTED, ChatChunk("Hel"), ChatChunk("lo")]
    assert body.closed


def test_a_full_stream_across_tasks_with_the_deadline_armed():
    provider = make_provider(Upstream(Body(normal_stream(), delay=0.01)), timeout_seconds=30)

    async def scenario():
        stream = provider.stream(chat_request())
        chunks = []
        while True:
            try:
                chunks.append(await asyncio.create_task(stream.__anext__()))
            except StopAsyncIteration:
                break
        await provider.aclose()
        return chunks

    assert asyncio.run(scenario())[-1] == ChatChunk("", "stop", Usage(5, 2))


@pytest.fixture
def loopback_upstream():
    """A real HTTP server on 127.0.0.1 that streams one event, then waits; it records when the
    client closes the connection."""
    state = {"closed_at": None, "connections": 0}

    async def handle(reader, writer):
        state["connections"] += 1
        while (await reader.readline()) not in (b"\r\n", b""):
            pass
        body = event(delta("hello"))
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
            + f"{len(body):x}\r\n".encode() + body + b"\r\n"
        )
        await writer.drain()
        await reader.read()  # returns b"" when the client closes the connection
        state["closed_at"] = time.monotonic()
        writer.close()

    return state, handle


def test_closing_a_real_stream_closes_the_upstream_connection(loopback_upstream):
    state, handle = loopback_upstream

    async def scenario():
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        provider = OpenAICompatibleProvider(f"http://127.0.0.1:{port}/v1", timeout_seconds=30, name="loopback")
        try:
            stream = provider.stream(chat_request())
            assert await stream.__anext__() == ACCEPTED
            assert await stream.__anext__() == ChatChunk("hello")
            closed = time.monotonic()
            await stream.aclose()
            for _ in range(100):
                if state["closed_at"] is not None:
                    break
                await asyncio.sleep(0.01)
            return closed
        finally:
            await provider.aclose()
            server.close()
            await server.wait_closed()

    closed = asyncio.run(scenario())
    assert state["closed_at"] is not None and state["closed_at"] - closed < 1
    assert state["connections"] == 1


# --- SSE decoder unit tests ---


def test_decoder_joins_data_and_dispatches_on_blank_lines():
    decoder = SSEDecoder()

    assert decoder.feed(b"data: a\ndata:b\ndata\n\n") == ["a\nb\n"]
    assert decoder.feed(b"data: c\r\n\r\ndata: d\r\r") == ["c", "d"]


def test_decoder_handles_crlf_split_across_reads():
    decoder = SSEDecoder()

    assert decoder.feed(b"data: a\r") == []
    assert decoder.feed(b"\n\r") == ["a"]  # the LF completes CRLF; the CR is the blank line
    assert decoder.feed(b"\ndata: b\n\n") == ["b"]


def test_decoder_keeps_only_one_leading_space_and_colons_in_values():
    assert SSEDecoder().feed(b"data:  two spaces: and colon\n\n") == [" two spaces: and colon"]


def test_decoder_close_discards_the_unfinished_event():
    decoder = SSEDecoder()
    decoder.feed(b"data: pending\n")
    decoder.close()

    assert decoder.feed(b"data: next\n\n") == ["next"]


def test_decoder_replaces_invalid_utf8():
    assert SSEDecoder().feed(b"data: \xff\n\n") == ["�"]


def test_decoder_limit_counts_the_whole_event():
    decoder = SSEDecoder(max_event_bytes=20)
    decoder.feed(b"data: 0123456789\n")  # 17 bytes

    with pytest.raises(SSEEventTooLargeError):
        decoder.feed(b"data: x\n")


def test_decoder_limit_resets_per_event():
    decoder = SSEDecoder(max_event_bytes=20)

    for _ in range(10):
        assert decoder.feed(b"data: 0123456789\n\n") == ["0123456789"]


# --- Configuration ---


def test_stream_usage_defaults_to_false_and_reaches_the_provider():
    default = OpenAICompatibleProviderConfig(type="openai_compatible", base_url=BASE_URL)
    enabled = OpenAICompatibleProviderConfig(type="openai_compatible", base_url=BASE_URL, stream_usage=True)

    assert default.stream_usage is False
    assert create_provider("a", default)._stream_usage is False
    assert create_provider("b", enabled)._stream_usage is True


def test_stream_usage_must_be_a_boolean_on_openai_compatible_providers_only():
    def config(provider):
        return json.dumps({"providers": {"p": provider}, "models": {"m": {"provider": "p", "provider_model_id": "m"}}})

    parse_registry_config(config({"type": "openai_compatible", "base_url": BASE_URL, "stream_usage": True}))
    with pytest.raises(RegistryConfigError, match="stream_usage"):
        parse_registry_config(config({"type": "openai_compatible", "base_url": BASE_URL, "stream_usage": "yes"}))
    with pytest.raises(RegistryConfigError, match="stream_usage"):
        parse_registry_config(config({"type": "fake", "stream_usage": True}))


def test_logs_never_contain_the_api_key(caplog):
    caplog.set_level(logging.DEBUG, logger="app")
    collect_error(Upstream(Body([event(delta("x"))], then_raise=httpx.ReadError("reset"))))
    collect(Upstream(Body(normal_stream(usage=False))))

    assert "sk-secret" not in caplog.text
