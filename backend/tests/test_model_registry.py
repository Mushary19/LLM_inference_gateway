"""Model registry: configuration validation, construction, resolution and lifecycle.

Pure unit tests: no database, no network. Environments are passed explicitly, so no real API
key is ever read.
"""

import asyncio
import copy
import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.core.config import BACKEND_DIR, Settings, settings
from app.database.connection import get_db
from app.main import app
from app.providers.base import ChatMessage, ChatRequest
from app.providers.fake import FAKE_MODEL, FakeProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.registry import (
    ModelRegistry,
    ModelRoute,
    RegistryConfigError,
    UnknownModelError,
    build_registry,
    default_registry,
    load_registry,
    parse_registry_config,
    registry_from_settings,
)

SECRET = "gsk_test_secret_value_0123456789"
ENV = {"GROQ_API_KEY": SECRET}

VALID = {
    "providers": {
        "groq": {
            "type": "openai_compatible",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
            "timeout_seconds": 60,
        },
        "ollama": {"type": "openai_compatible", "base_url": "http://localhost:11434/v1", "timeout_seconds": 120},
        "fake": {"type": "fake"},
    },
    "models": {
        "llama-3.3-70b": {"provider": "groq", "provider_model_id": "llama-3.3-70b-versatile"},
        "llama-3.1-8b": {"provider": "groq", "provider_model_id": "llama-3.1-8b-instant"},
        "qwen-local": {"provider": "ollama", "provider_model_id": "qwen2.5:7b", "enabled": False},
        "fake-model": {"provider": "fake", "provider_model_id": "fake-model"},
    },
}


def config(**changes):
    """A deep copy of VALID; `changes` maps dotted paths to new values (None deletes).
    Path parts can't contain dots, so IDs like "llama-3.1-8b" are edited directly."""
    data = copy.deepcopy(VALID)
    for path, value in changes.items():
        *parents, last = path.split(".")
        target = data
        for part in parents:
            target = target[part]
        if value is None:
            del target[last]
        else:
            target[last] = value
    return data


def build(data, environ=ENV) -> ModelRegistry:
    return build_registry(parse_registry_config(json.dumps(data)), environ)


def config_error(data, environ=ENV) -> str:
    with pytest.raises(RegistryConfigError) as excinfo:
        registry = build(data, environ)
        asyncio.run(registry.aclose())  # pragma: no cover - only reached if no error was raised
    return str(excinfo.value)


def close(registry: ModelRegistry) -> None:
    asyncio.run(registry.aclose())


# --- Valid configuration ---


def test_valid_configuration_builds_routes():
    registry = build(VALID)
    try:
        route = registry.resolve("llama-3.3-70b")
        assert route == ModelRoute("llama-3.3-70b", route.provider, "llama-3.3-70b-versatile")
        assert isinstance(route.provider, OpenAICompatibleProvider) and route.provider.name == "groq"
        assert route.provider._client.headers["Authorization"] == f"Bearer {SECRET}"
        assert str(route.provider._client.base_url) == "https://api.groq.com/openai/v1/"

        fake = registry.resolve("fake-model")
        assert isinstance(fake.provider, FakeProvider) and fake.provider.name == "fake"
        assert fake.provider_model_id == "fake-model"
    finally:
        close(registry)


def test_each_provider_instance_is_configured_separately():
    registry = build(config(**{"models.qwen-local.enabled": True}))
    try:
        groq = registry.resolve("llama-3.3-70b").provider
        ollama = registry.resolve("qwen-local").provider
        assert groq is not ollama
        assert (groq.name, ollama.name) == ("groq", "ollama")
        assert str(ollama._client.base_url) == "http://localhost:11434/v1/"
        assert ollama._client.timeout.read == 120 and groq._client.timeout.read == 60
        assert "Authorization" not in ollama._client.headers  # no api_key_env: no key sent
    finally:
        close(registry)


def test_models_on_one_provider_share_its_instance_and_connection_pool():
    registry = build(VALID)
    try:
        first = registry.resolve("llama-3.3-70b").provider
        second = registry.resolve("llama-3.1-8b").provider
        assert first is second
        assert first._client is second._client
    finally:
        close(registry)


def test_enabled_defaults_to_true_and_timeout_has_a_default():
    registry = build(config(**{"providers.ollama.timeout_seconds": None, "models.qwen-local.enabled": None}))
    try:
        assert registry.resolve("qwen-local").provider._client.timeout.read == 60
    finally:
        close(registry)


def test_the_committed_example_is_valid():
    registry = load_registry(BACKEND_DIR / "models.example.json", ENV)
    try:
        assert [r.public_id for r in registry.enabled_models()] == ["fake-model", "llama-3.3-70b"]
        with pytest.raises(UnknownModelError):
            registry.resolve("qwen-local")
    finally:
        close(registry)


def test_provider_model_ids_may_use_upstream_naming():
    ids = ["qwen2.5:7b", "accounts/fireworks/models/llama-v3p1-8b", "meta-llama/Llama-3.3-70B-Instruct", "a@b+c"]
    data = config(models={f"m{i}": {"provider": "fake", "provider_model_id": pid} for i, pid in enumerate(ids)})
    registry = build(data)
    try:
        assert [registry.resolve(f"m{i}").provider_model_id for i in range(len(ids))] == ids
    finally:
        close(registry)


# --- Invalid configuration ---


def test_invalid_json_is_rejected_with_its_position():
    with pytest.raises(RegistryConfigError, match=r"invalid JSON at line 1, column \d+"):
        parse_registry_config('{"providers": {,}')


@pytest.mark.parametrize(
    "text, key",
    [
        ('{"providers": {}, "providers": {}, "models": {}}', "providers"),
        ('{"providers": {"fake": {"type": "fake"}, "fake": {"type": "fake"}}, "models": {}}', "fake"),
        (
            '{"providers": {"fake": {"type": "fake"}}, "models": {'
            '"m": {"provider": "fake", "provider_model_id": "a"}, "m": {"provider": "fake", "provider_model_id": "b"}}}',
            "m",
        ),
        (
            '{"providers": {"fake": {"type": "fake"}}, "models": {'
            '"m": {"provider": "fake", "provider_model_id": "a", "provider_model_id": "b"}}}',
            "provider_model_id",
        ),
    ],
    ids=["top-level", "provider", "model", "field"],
)
def test_duplicate_json_keys_are_rejected(text, key):
    with pytest.raises(RegistryConfigError, match=f"duplicate key '{key}'"):
        parse_registry_config(text)


@pytest.mark.parametrize(
    "data, location",
    [
        (config(extra={}), "extra"),
        (config(**{"providers.groq.api_key": "sk-inline"}), "providers.groq.openai_compatible.api_key"),
        (config(**{"providers.fake.base_url": "http://x"}), "providers.fake.fake.base_url"),
        (config(**{"models.fake-model.weight": 2}), "models.fake-model.weight"),
    ],
    ids=["top-level", "provider", "fake-provider", "model"],
)
def test_unknown_fields_are_rejected(data, location):
    message = config_error(data)

    assert f"{location}: Extra inputs are not permitted" in message
    assert "sk-inline" not in message


def test_unknown_provider_type_is_rejected():
    message = config_error(config(**{"providers.groq.type": "anthropic_native"}))

    assert "providers.groq" in message and "does not match any of the expected tags" in message


def test_missing_provider_type_is_rejected():
    assert "providers.fake" in config_error(config(**{"providers.fake.type": None}))


def test_missing_provider_reference_is_rejected():
    data = config()
    data["models"]["llama-3.1-8b"]["provider"] = "openai"  # (IDs with dots don't fit config()'s paths)

    message = config_error(data)

    assert "models.llama-3.1-8b.provider: unknown provider 'openai'" in message


@pytest.mark.parametrize(
    "path",
    ["providers", "models", "models.fake-model.provider", "models.fake-model.provider_model_id", "providers.groq.base_url"],
)
def test_missing_required_settings_are_rejected(path):
    location = path.replace("providers.groq.", "providers.groq.openai_compatible.")
    assert f"{location}: Field required" in config_error(config(**{path: None}))


def test_at_least_one_model_is_required():
    assert "models" in config_error(config(models={}))


def test_top_level_must_be_an_object():
    assert "(top level)" in config_error([])


@pytest.mark.parametrize(
    "base_url",
    ["", "localhost:11434/v1", "ftp://example.com/v1", "http://", "http://[broken/v1"],
)
def test_invalid_base_urls_are_rejected(base_url):
    message = config_error(config(**{"providers.ollama.base_url": base_url}))

    assert "providers.ollama.openai_compatible.base_url" in message
    assert "http:// or https://" in message


def test_base_url_with_embedded_credentials_is_rejected_without_echoing_them():
    message = config_error(config(**{"providers.ollama.base_url": "https://admin:hunter2@llm.internal/v1"}))

    assert "must not contain credentials" in message
    assert "hunter2" not in message and "admin" not in message and "llm.internal" not in message


@pytest.mark.parametrize(
    "path, value",
    [
        ("models.qwen-local.enabled", "false"),
        ("providers.ollama.timeout_seconds", "60"),
        ("providers.ollama.timeout_seconds", 0),
        ("providers.ollama.timeout_seconds", -5),
        ("providers.ollama.timeout_seconds", True),
        ("models.fake-model.provider_model_id", 7),
    ],
)
def test_values_are_type_checked_strictly(path, value):
    assert path.split(".")[-1] in config_error(config(**{path: value}))


def test_missing_environment_variable_is_rejected_by_name():
    message = config_error(VALID, environ={})

    assert "providers.groq.api_key_env: environment variable GROQ_API_KEY is not set or empty" in message


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_environment_variable_is_rejected(value):
    assert "GROQ_API_KEY" in config_error(VALID, environ={"GROQ_API_KEY": value})


def test_every_missing_variable_is_reported_at_once():
    data = config(**{"providers.ollama.api_key_env": "OLLAMA_TOKEN"})

    message = config_error(data, environ={"UNRELATED": SECRET})

    assert "GROQ_API_KEY" in message and "OLLAMA_TOKEN" in message
    assert SECRET not in message


def test_api_key_is_read_from_the_named_variable_only():
    data = config(**{"providers.groq.api_key_env": "MY_GROQ_KEY"})
    registry = build(data, environ={"MY_GROQ_KEY": " other-secret \n", "GROQ_API_KEY": SECRET})
    try:
        headers = registry.resolve("llama-3.3-70b").provider._client.headers
        assert headers["Authorization"] == "Bearer other-secret"
    finally:
        close(registry)


@pytest.mark.parametrize(
    "data, location",
    [
        (config(models={"bad id": {"provider": "fake", "provider_model_id": "x"}}), "models.bad id.[key]"),
        (config(models={"-leading": {"provider": "fake", "provider_model_id": "x"}}), "models.-leading.[key]"),
        (config(models={"": {"provider": "fake", "provider_model_id": "x"}}), "models..[key]"),
        (config(models={"m" * 129: {"provider": "fake", "provider_model_id": "x"}}), "[key]"),
        (config(models={"emoji-🙂": {"provider": "fake", "provider_model_id": "x"}}), "[key]"),
        (config(**{"providers.Groq": {"type": "fake"}}), "providers.Groq.[key]"),
        (config(**{"providers.my provider": {"type": "fake"}}), "providers.my provider.[key]"),
        (config(**{"models.fake-model.provider_model_id": "has space"}), "models.fake-model.provider_model_id"),
        (config(**{"models.fake-model.provider_model_id": ""}), "models.fake-model.provider_model_id"),
        (config(**{"models.fake-model.provider_model_id": "tab\there"}), "models.fake-model.provider_model_id"),
        (config(**{"providers.groq.api_key_env": "GROQ-KEY"}), "providers.groq.openai_compatible.api_key_env"),
    ],
    ids=[
        "space", "leading-dash", "empty", "too-long", "non-ascii", "upper-case-provider",
        "provider-space", "upstream-space", "upstream-empty", "upstream-control", "env-var-name",
    ],
)
def test_malformed_ids_are_rejected(data, location):
    assert location in config_error(data)


def test_missing_registry_file_is_reported(tmp_path):
    with pytest.raises(RegistryConfigError, match="cannot read model registry file .*missing.json"):
        load_registry(tmp_path / "missing.json", ENV)


def test_errors_name_the_file(tmp_path):
    path = tmp_path / "models.json"
    path.write_text("{}")

    with pytest.raises(RegistryConfigError, match=f"^{path}: invalid configuration"):
        load_registry(path, ENV)


# --- Resolution and listing ---


def test_unknown_and_disabled_models_raise_unknown_model_error():
    registry = build(VALID)
    try:
        with pytest.raises(UnknownModelError) as unknown:
            registry.resolve("gpt-4o")
        with pytest.raises(UnknownModelError) as disabled:
            registry.resolve("qwen-local")
    finally:
        close(registry)

    assert (unknown.value.public_id, unknown.value.disabled) == ("gpt-4o", False)
    assert (disabled.value.public_id, disabled.value.disabled) == ("qwen-local", True)
    assert "disabled" in str(disabled.value) and "not registered" in str(unknown.value)


@pytest.mark.parametrize("model", ["LLAMA-3.3-70B", "Llama-3.3-70b", "llama-3.3-70b ", " llama-3.3-70b", "llama-3.3"])
def test_resolution_is_exact_and_case_sensitive(model):
    registry = build(VALID)
    try:
        with pytest.raises(UnknownModelError):
            registry.resolve(model)
    finally:
        close(registry)


def test_provider_model_ids_are_not_public_ids():
    registry = build(VALID)
    try:
        with pytest.raises(UnknownModelError):
            registry.resolve("llama-3.3-70b-versatile")
    finally:
        close(registry)


def test_listing_is_sorted_stable_and_excludes_disabled_models():
    shuffled = config(models=dict(reversed(list(VALID["models"].items()))))
    first, second = build(VALID), build(shuffled)
    try:
        expected = ["fake-model", "llama-3.1-8b", "llama-3.3-70b"]
        assert [r.public_id for r in first.enabled_models()] == expected
        assert [r.public_id for r in second.enabled_models()] == expected
        assert first.enabled_models() == first.enabled_models()
    finally:
        close(first)
        close(second)


def test_registry_cannot_be_changed_after_construction():
    registry = build(VALID)
    try:
        with pytest.raises(TypeError):
            registry._routes["new"] = registry.resolve("fake-model")  # read-only mapping
        with pytest.raises(AttributeError):
            registry.resolve("fake-model").provider_model_id = "other"  # frozen route
        listed = registry.enabled_models()
        listed.clear()  # a copy: the registry is unaffected
        assert len(registry.enabled_models()) == 3
    finally:
        close(registry)


def test_registry_rejects_inconsistent_routes():
    owned, foreign = FakeProvider(name="owned"), FakeProvider(name="foreign")
    with pytest.raises(ValueError, match="duplicate public model ID"):
        ModelRegistry({"owned": owned}, [ModelRoute("m", owned, "x"), ModelRoute("m", owned, "y")])
    with pytest.raises(ValueError, match="does not own"):
        ModelRegistry({"owned": owned}, [ModelRoute("m", foreign, "x")])
    with pytest.raises(ValueError, match="both enabled and disabled"):
        ModelRegistry({"owned": owned}, [ModelRoute("m", owned, "x")], disabled=["m"])


# --- Secrets ---


def test_secrets_never_appear_in_reprs():
    registry = build(VALID)
    try:
        route = registry.resolve("llama-3.3-70b")
        for text in (repr(registry), repr(route), repr(route.provider), str(registry.enabled_models())):
            assert SECRET not in text
            assert "api.groq.com" not in text  # base URLs stay internal too
        assert repr(registry) == (
            "ModelRegistry(models=['fake-model', 'llama-3.1-8b', 'llama-3.3-70b'], providers=['fake', 'groq', 'ollama'])"
        )
    finally:
        close(registry)


def test_validation_errors_never_echo_values_from_the_file():
    data = config(**{
        "providers.groq.api_key": SECRET,  # an inline key is an unknown field: rejected, not echoed
        "providers.ollama.timeout_seconds": "secret-looking-string",
    })

    message = config_error(data)

    assert SECRET not in message and "secret-looking-string" not in message


def test_validation_errors_are_not_chained_to_the_raw_pydantic_error():
    with pytest.raises(RegistryConfigError) as excinfo:
        parse_registry_config(json.dumps(config(**{"providers.groq.api_key": SECRET})))

    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


# --- Lifecycle ---


class ClosingProvider(FakeProvider):
    def __init__(self, name, fail=False, log=None):
        super().__init__(name=name)
        self.fail = fail
        self.log = log

    async def aclose(self):
        self.log.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")


def test_aclose_closes_every_provider_even_if_one_fails(caplog):
    log: list[str] = []
    providers = {name: ClosingProvider(name, fail=(name == "b"), log=log) for name in ("a", "b", "c")}
    registry = ModelRegistry(providers, [ModelRoute("m", providers["a"], FAKE_MODEL)])

    with caplog.at_level(logging.ERROR):
        asyncio.run(registry.aclose())  # does not raise

    assert log == ["a", "b", "c"]  # c was still closed after b failed
    assert "Failed to close provider b" in caplog.text


def test_aclose_closes_http_clients():
    registry = build(VALID)
    provider = registry.resolve("llama-3.3-70b").provider

    close(registry)

    assert provider._client.is_closed


# --- Default registry and settings ---


def test_default_registry_preserves_the_fake_model():
    registry = default_registry()

    [route] = registry.enabled_models()
    assert (route.public_id, route.provider_model_id) == (FAKE_MODEL, FAKE_MODEL)
    assert isinstance(route.provider, FakeProvider) and route.provider.name == "fake"
    response = asyncio.run(route.provider.chat(ChatRequest(FAKE_MODEL, (ChatMessage("user", "hi"),))))
    assert response.message.content == "Fake response to: hi"


def test_without_a_registry_file_the_default_registry_is_used(monkeypatch):
    monkeypatch.setattr(settings, "model_registry_file", None)
    monkeypatch.setattr(settings, "chat_provider", "fake")

    registry = registry_from_settings(settings, environ={})

    assert [r.public_id for r in registry.enabled_models()] == [FAKE_MODEL]


@pytest.mark.parametrize("legacy", ["openai_compatible", "something-else"])
def test_legacy_real_provider_without_a_registry_file_fails_with_migration_help(monkeypatch, legacy):
    monkeypatch.setattr(settings, "model_registry_file", None)
    monkeypatch.setattr(settings, "chat_provider", legacy)

    with pytest.raises(RegistryConfigError) as excinfo:
        registry_from_settings(settings, environ={})

    message = str(excinfo.value)
    assert f"CHAT_PROVIDER={legacy!r} is no longer supported" in message
    assert "MODEL_REGISTRY_FILE" in message and "models.example.json" in message


def test_registry_file_wins_over_legacy_settings_with_a_warning(monkeypatch, tmp_path, caplog):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(VALID))
    monkeypatch.setattr(settings, "model_registry_file", path)
    monkeypatch.setattr(settings, "chat_provider", "openai_compatible")

    registry = registry_from_settings(settings, environ=ENV)
    try:
        assert registry.resolve("llama-3.3-70b").provider.name == "groq"
    finally:
        close(registry)
    assert "CHAT_PROVIDER is ignored" in caplog.text
    assert SECRET not in caplog.text


def test_application_refuses_to_start_with_legacy_settings(monkeypatch):
    monkeypatch.setattr(settings, "chat_provider", "openai_compatible")

    with pytest.raises(RegistryConfigError, match="no longer supported"):
        with TestClient(app):
            pass  # pragma: no cover


def test_application_refuses_to_start_with_an_invalid_registry(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(config(**{"models.fake-model.provider": "nope"})))
    monkeypatch.setattr(settings, "model_registry_file", path)

    with pytest.raises(RegistryConfigError, match="unknown provider 'nope'"):
        with TestClient(app):
            pass  # pragma: no cover


def test_application_serves_the_configured_registry(monkeypatch, tmp_path, db_session, auth_headers):
    """GET /v1/models lists what the file registers; no provider is called."""
    path = tmp_path / "models.json"
    path.write_text(json.dumps(config(**{"providers.groq.api_key_env": None})))
    monkeypatch.setattr(settings, "model_registry_file", path)
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app, headers=auth_headers) as client:
            listed = client.get("/v1/models").json()
            assert repr(app.state.model_registry).startswith("ModelRegistry(models=['fake-model'")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert [m["id"] for m in listed["data"]] == ["fake-model", "llama-3.1-8b", "llama-3.3-70b"]


@pytest.mark.parametrize(
    "value, expected",
    [
        ("models.json", BACKEND_DIR / "models.json"),
        ("config/models.json", BACKEND_DIR / "config" / "models.json"),
        ("/etc/inferenceflow/models.json", "/etc/inferenceflow/models.json"),
        ("", None),
        ("   ", None),
    ],
)
def test_registry_path_resolution(value, expected):
    configured = Settings(database_url=settings.database_url, model_registry_file=value)

    assert configured.model_registry_file == (None if expected is None else type(BACKEND_DIR)(expected))


# --- Read access for diagnostics ---


def test_disabled_routes_are_kept_for_diagnostics_but_never_resolved():
    registry = build(VALID)
    try:
        [disabled] = registry.disabled_models()
        assert (disabled.public_id, disabled.provider.name, disabled.provider_model_id) == ("qwen-local", "ollama", "qwen2.5:7b")
        with pytest.raises(UnknownModelError) as excinfo:
            registry.resolve("qwen-local")
        assert excinfo.value.disabled
        assert "qwen-local" not in [r.public_id for r in registry.enabled_models()]
    finally:
        close(registry)


def test_providers_accessor_is_read_only_and_includes_every_instance():
    registry = build(VALID)
    try:
        assert sorted(registry.providers) == ["fake", "groq", "ollama"]  # ollama serves only a disabled model
        with pytest.raises(TypeError):
            registry.providers["new"] = FakeProvider()
    finally:
        close(registry)


def test_disabled_models_may_still_be_given_as_ids():
    provider = FakeProvider()
    registry = ModelRegistry({"fake": provider}, [ModelRoute(FAKE_MODEL, provider, FAKE_MODEL)], disabled=["old"])

    with pytest.raises(UnknownModelError) as excinfo:
        registry.resolve("old")
    assert excinfo.value.disabled and registry.disabled_models() == []


def test_disabled_routes_must_use_owned_providers():
    owned, foreign = FakeProvider(name="owned"), FakeProvider(name="foreign")

    with pytest.raises(ValueError, match="does not own"):
        ModelRegistry({"owned": owned}, [], disabled=[ModelRoute("m", foreign, "x")])
