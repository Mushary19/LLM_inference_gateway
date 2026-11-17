"""python -m app.cli check-providers: output, exit status, isolation from the database,
provider cleanup, and real loopback failures (refused and stalled connections)."""

import io
import json
import socket
import threading
import time

import httpx2 as httpx
import pytest

from app import cli
from app.core.config import settings
from app.providers.fake import FAKE_MODEL, FakeProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.registry import ModelRegistry, ModelRoute

SECRET = "sk-cli-secret-0123456789"
BASE_URL = "http://internal-llm.corp.example:8000/v1"


def no_database():
    raise AssertionError("check-providers must not open a database session")


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["check-providers", *argv], session_scope=no_database, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def mock_provider(name, handler):
    return OpenAICompatibleProvider(BASE_URL, api_key=SECRET, transport=httpx.MockTransport(handler), name=name)


def listing(*ids):
    return lambda request: httpx.Response(200, json={"data": [{"id": i, "owned_by": "x"} for i in ids]})


@pytest.fixture
def use_registry(monkeypatch):
    """Make the CLI load this registry instead of the configured one; returns it for inspection."""
    holder = {}

    def install(registry):
        holder["registry"] = registry
        monkeypatch.setattr(cli, "registry_from_settings", lambda _settings: registry)
        return registry

    return install


def mixed_registry():
    groq = mock_provider("groq", listing("llama-3.3-70b-versatile"))
    broken = mock_provider("broken", lambda request: httpx.Response(503, text=f"down at {BASE_URL} {SECRET}"))
    fake = FakeProvider(name="fake")
    return ModelRegistry(
        {"groq": groq, "broken": broken, "fake": fake},
        [
            ModelRoute("llama-3.3-70b", groq, "llama-3.3-70b-versatile"),
            ModelRoute("llama-3.1-8b", groq, "llama-3.1-8b-instnt"),
            ModelRoute(FAKE_MODEL, fake, FAKE_MODEL),
        ],
        disabled=[ModelRoute("qwen-local", broken, "qwen2.5:7b")],
    )


# --- Output and exit status ---


def test_everything_healthy_exits_0(use_registry):
    groq = mock_provider("groq", listing("llama-3.3-70b-versatile"))
    use_registry(ModelRegistry({"groq": groq}, [ModelRoute("llama-3.3-70b", groq, "llama-3.3-70b-versatile")]))

    code, out, err = run()

    assert code == 0 and err == ""
    assert "No problems found." in out
    assert "groq      healthy" in out


def test_problems_exit_1_with_a_readable_report(use_registry):
    use_registry(mixed_registry())

    code, out, err = run()

    assert code == 1 and err == ""
    lines = out.splitlines()
    assert lines[0] == "Registry: built-in default (MODEL_REGISTRY_FILE is not set)"
    providers = lines[lines.index(next(l for l in lines if l.startswith("PROVIDER"))) + 1:][:3]
    assert [line.split()[:3] for line in providers] == [
        ["broken", "unhealthy", "server_error"],
        ["fake", "healthy", "-"],
        ["groq", "healthy", "-"],
    ]
    models = lines[lines.index(next(l for l in lines if l.startswith("MODEL"))) + 1:][:4]
    assert [line.split() for line in models] == [
        [FAKE_MODEL, "fake", FAKE_MODEL, "yes", "listed"],
        ["llama-3.1-8b", "groq", "llama-3.1-8b-instnt", "yes", "not_listed"],
        ["llama-3.3-70b", "groq", "llama-3.3-70b-versatile", "yes", "listed"],
        ["qwen-local", "broken", "qwen2.5:7b", "no", "unknown"],
    ]
    assert "  provider broken is unhealthy (server_error)" in lines
    assert "  enabled model llama-3.1-8b is not_listed on provider groq" in lines
    assert "requests may still work" in out  # not_listed is never presented as definitive


def test_json_report(use_registry):
    use_registry(mixed_registry())

    code, out, _ = run("--json")

    report = json.loads(out)
    assert code == 1 and report["ok"] is False and report["registry"] is None
    assert report["providers"][0] == {
        "name": "broken", "status": "unhealthy", "reason": "server_error", "latency_ms": report["providers"][0]["latency_ms"],
    }
    assert {m["id"]: m["availability"] for m in report["models"]} == {
        FAKE_MODEL: "listed", "llama-3.1-8b": "not_listed", "llama-3.3-70b": "listed", "qwen-local": "unknown",
    }
    assert set(report["models"][0]) == {"id", "provider", "provider_model_id", "enabled", "availability"}
    assert len(report["problems"]) == 2


@pytest.mark.parametrize("flag", [[], ["--json"]])
def test_output_never_contains_credentials_or_urls(use_registry, flag):
    use_registry(mixed_registry())

    _, out, err = run(*flag)

    for leaked in (SECRET, "internal-llm", "corp.example", "8000", "down at"):
        assert leaked not in out and leaked not in err


def test_the_built_in_registry_is_checked_when_no_file_is_configured():
    code, out, _ = run("--json")

    report = json.loads(out)
    assert code == 0 and report["ok"]
    assert report["providers"] == [{"name": "fake", "status": "healthy", "reason": None, "latency_ms": report["providers"][0]["latency_ms"]}]
    assert [(m["id"], m["availability"]) for m in report["models"]] == [(FAKE_MODEL, "listed")]


# --- Configuration errors exit 2 ---


def test_legacy_settings_exit_2(monkeypatch):
    monkeypatch.setattr(settings, "chat_provider", "openai_compatible")

    code, out, err = run()

    assert code == 2 and out == ""
    assert "Model registry configuration error" in err and "MODEL_REGISTRY_FILE" in err


def test_invalid_registry_file_exits_2(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    path.write_text('{"providers": {}, "providers": {}}')
    monkeypatch.setattr(settings, "model_registry_file", path)

    code, _, err = run()

    assert code == 2 and "duplicate key 'providers'" in err


def test_missing_api_key_variable_exits_2_without_probing(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "providers": {"groq": {"type": "openai_compatible", "base_url": BASE_URL, "api_key_env": "GROQ_API_KEY"}},
        "models": {"m": {"provider": "groq", "provider_model_id": "x"}},
    }))
    monkeypatch.setattr(settings, "model_registry_file", path)

    code, _, err = run()

    assert code == 2 and "GROQ_API_KEY" in err


@pytest.mark.parametrize("value", ["0", "-1", "abc", "301", "nan"])
def test_invalid_timeouts_are_rejected(value, capsys):
    with pytest.raises(SystemExit) as excinfo:
        run("--timeout", value)

    assert excinfo.value.code == 2
    assert "--timeout" in capsys.readouterr().err


def test_largest_timeout_is_accepted(use_registry):
    use_registry(ModelRegistry({"fake": (p := FakeProvider())}, [ModelRoute(FAKE_MODEL, p, FAKE_MODEL)]))

    assert run("--timeout", "300")[0] == 0


# --- Database independence and cleanup ---


def test_other_commands_still_open_a_session():
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(AssertionError, match="must not open a database session"):
        cli.main(["list-keys"], session_scope=no_database, out=out, err=err)


def test_providers_are_closed_after_a_run(use_registry):
    registry = use_registry(mixed_registry())

    run()

    assert registry.providers["groq"]._client.is_closed
    assert registry.providers["broken"]._client.is_closed


def test_providers_are_closed_when_checking_itself_fails(use_registry, monkeypatch):
    registry = use_registry(mixed_registry())

    async def failing_check(*args, **kwargs):
        raise RuntimeError("diagnostics bug")

    monkeypatch.setattr(cli, "check_registry", failing_check)

    with pytest.raises(RuntimeError, match="diagnostics bug"):
        run()

    assert registry.providers["groq"]._client.is_closed
    assert registry.providers["broken"]._client.is_closed


def test_a_failing_close_does_not_hide_the_report(use_registry, caplog):
    class UnclosableProvider(FakeProvider):
        async def aclose(self):
            raise RuntimeError("close failed")

    groq = mock_provider("groq", listing("m"))
    bad = UnclosableProvider(name="bad")
    use_registry(ModelRegistry({"bad": bad, "groq": groq}, [ModelRoute("m", groq, "m"), ModelRoute(FAKE_MODEL, bad, FAKE_MODEL)]))

    code, out, _ = run()

    assert code == 0 and "No problems found." in out
    assert groq._client.is_closed  # still closed after "bad" failed to close
    assert "Failed to close provider bad" in caplog.text


# --- Real loopback failures (no mocks below the HTTP client) ---


@pytest.fixture
def closed_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def stalled_port():
    """A server that accepts connections and never answers."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    server.settimeout(0.1)
    held, stop = [], threading.Event()

    def accept():
        while not stop.is_set():
            try:
                held.append(server.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    thread.join()
    for connection in held:
        connection.close()
    server.close()


def registry_file(tmp_path, ports):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "providers": {
            name: {"type": "openai_compatible", "base_url": f"http://127.0.0.1:{port}/v1", "timeout_seconds": 120}
            for name, port in ports.items()
        } | {"fake": {"type": "fake"}},
        "models": {f"{name}-model": {"provider": name, "provider_model_id": "m"} for name in ports}
        | {FAKE_MODEL: {"provider": "fake", "provider_model_id": FAKE_MODEL}},
    }))
    return path


def test_real_refused_and_stalled_connections(monkeypatch, tmp_path, closed_port, stalled_port):
    path = registry_file(tmp_path, {"refused": closed_port, "stalled": stalled_port})
    monkeypatch.setattr(settings, "model_registry_file", path)

    started = time.monotonic()
    code, out, _ = run("--json", "--timeout", "0.5")
    elapsed = time.monotonic() - started

    report = json.loads(out)
    assert code == 1 and report["registry"] == str(path)
    assert {p["name"]: (p["status"], p["reason"]) for p in report["providers"]} == {
        "fake": ("healthy", None),
        "refused": ("unhealthy", "connection_failed"),
        "stalled": ("unhealthy", "timeout"),  # the 120 s inference timeout doesn't apply
    }
    assert {m["id"]: m["availability"] for m in report["models"]} == {
        FAKE_MODEL: "listed", "refused-model": "unknown", "stalled-model": "unknown",
    }
    assert elapsed < 5
    assert "127.0.0.1" not in out and str(stalled_port) not in out
