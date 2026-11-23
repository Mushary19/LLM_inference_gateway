"""Model registry: maps public model IDs to a provider instance and that provider's model ID.

Clients only ever see public IDs. Example: the public ID "llama-3.3-70b" might route to the
provider instance "groq" (an openai_compatible provider) as "llama-3.3-70b-versatile".

The registry is loaded once at startup from a JSON file (settings.model_registry_file), fully
validated, and immutable afterwards: changing it requires a restart. API keys never appear
in the file. A provider names the environment variable that holds its key (`api_key_env`),
and the key goes straight into that provider's HTTP client. It is never stored on the
configuration, logged, or put in an error message.

Configuration format (see models.example.json):

    {
      "providers": {"<provider name>": {"type": "openai_compatible", "base_url": "...",
                                        "api_key_env": "...", "timeout_seconds": 60,
                                        "stream_usage": false},
                    "<provider name>": {"type": "fake"}},
      "models":    {"<public model ID>": {"provider": "<provider name>",
                                          "provider_model_id": "...", "enabled": true}}
    }
"""

import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Union
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator

from app.core.config import ENV_FILE, Settings
from app.providers.base import Provider
from app.providers.factory import create_provider
from app.providers.fake import FAKE_MODEL

logger = logging.getLogger(__name__)

# Public model IDs: what clients send. Letters, digits and . _ : / - (e.g. "llama-3.3-70b").
PublicModelId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")]
# Provider instance names: lower-case identifiers (e.g. "groq", "ollama-local").
ProviderName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
# Whatever the upstream calls the model ("qwen2.5:7b", "accounts/x/models/y"): any characters
# except whitespace and control characters.
ProviderModelId = Annotated[str, StringConstraints(pattern=r"^[^\s\x00-\x1f\x7f]{1,256}$")]
EnvVarName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")]


class RegistryConfigError(Exception):
    """The model registry configuration is invalid. Messages never contain secret values."""


class UnknownModelError(Exception):
    """The public model ID is not registered, or is registered but disabled."""

    def __init__(self, public_id: str, *, disabled: bool) -> None:
        super().__init__(f"model {public_id!r} is {'disabled' if disabled else 'not registered'}")
        self.public_id = public_id
        self.disabled = disabled


# --- Configuration schema ---


class _StrictModel(BaseModel):
    # Unknown fields are errors (so typos surface), and values are never coerced
    # ("true" is not a bool, 60 is fine for a float but "60" is not).
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class FakeProviderConfig(_StrictModel):
    """The deterministic in-process provider (app/providers/fake.py). It serves "fake-model" only."""

    type: Literal["fake"]


class OpenAICompatibleProviderConfig(_StrictModel):
    type: Literal["openai_compatible"]
    base_url: str
    # Name of the environment variable holding the bearer token, if the server needs one.
    api_key_env: EnvVarName | None = None
    # Covers the whole request, including generation (for streams: the whole stream).
    timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    # Ask the server for token usage on streamed responses (stream_options.include_usage).
    # Off by default because not every OpenAI-compatible server accepts that option; without
    # it, streams from servers that only report usage on request are charged the full
    # reservation. Enable it only after verifying the server supports it.
    stream_usage: bool = False

    @field_validator("base_url")
    @classmethod
    def _http_url_without_credentials(cls, value: str) -> str:
        try:
            parts = urlsplit(value)
        except ValueError:
            raise ValueError("must be a valid http:// or https:// URL") from None
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("must be a valid http:// or https:// URL")
        if parts.username is not None or parts.password is not None:
            raise ValueError("must not contain credentials; put the key in the variable named by api_key_env")
        return value


ProviderConfig = Annotated[Union[FakeProviderConfig, OpenAICompatibleProviderConfig], Field(discriminator="type")]


class ModelConfig(_StrictModel):
    provider: ProviderName
    provider_model_id: ProviderModelId
    enabled: bool = True


class RegistryConfig(_StrictModel):
    providers: dict[ProviderName, ProviderConfig]
    models: dict[PublicModelId, ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _models_reference_known_providers(self) -> "RegistryConfig":
        unknown = [
            f"models.{public_id}.provider: unknown provider {model.provider!r}"
            for public_id, model in self.models.items()
            if model.provider not in self.providers
        ]
        if unknown:
            raise ValueError("; ".join(unknown))
        return self


# --- Runtime registry ---


@dataclass(frozen=True)
class ModelRoute:
    """Where requests for one public model go."""

    public_id: str
    provider: Provider
    provider_model_id: str


class ModelRegistry:
    """Immutable mapping of public model IDs to routes. Owns its provider instances.

    Lookups are exact and case-sensitive. Disabled models behave exactly like unknown ones
    for callers. The registry still remembers them, so the error can say which it was, and
    keeps their routes (when given) so diagnostics can check them before they're enabled.
    """

    def __init__(
        self,
        providers: Mapping[str, Provider],
        routes: Iterable[ModelRoute],
        disabled: Iterable[str | ModelRoute] = (),
    ) -> None:
        self._providers: Mapping[str, Provider] = MappingProxyType(dict(providers))
        owned = {id(provider) for provider in self._providers.values()}

        def check_owned(route: ModelRoute) -> None:
            if id(route.provider) not in owned:
                raise ValueError(f"model {route.public_id!r} uses a provider this registry does not own")

        by_id: dict[str, ModelRoute] = {}
        for route in routes:
            if route.public_id in by_id:
                raise ValueError(f"duplicate public model ID {route.public_id!r}")
            check_owned(route)
            by_id[route.public_id] = route
        self._routes: Mapping[str, ModelRoute] = MappingProxyType(by_id)

        disabled_routes: dict[str, ModelRoute] = {}
        disabled_ids: set[str] = set()
        for item in disabled:
            if isinstance(item, ModelRoute):
                check_owned(item)
                disabled_routes[item.public_id] = item
                disabled_ids.add(item.public_id)
            else:
                disabled_ids.add(item)
        self._disabled = frozenset(disabled_ids)
        self._disabled_routes: Mapping[str, ModelRoute] = MappingProxyType(disabled_routes)
        if self._disabled & by_id.keys():
            raise ValueError("a model cannot be both enabled and disabled")

    def resolve(self, public_id: str) -> ModelRoute:
        route = self._routes.get(public_id)
        if route is None:
            raise UnknownModelError(public_id, disabled=public_id in self._disabled)
        return route

    def enabled_models(self) -> list[ModelRoute]:
        """Enabled routes, sorted by public ID so listings are stable."""
        return sorted(self._routes.values(), key=lambda route: route.public_id)

    def disabled_models(self) -> list[ModelRoute]:
        """Routes of disabled models, sorted by public ID. Never used for requests."""
        return sorted(self._disabled_routes.values(), key=lambda route: route.public_id)

    @property
    def providers(self) -> Mapping[str, Provider]:
        """Every provider instance by name (read-only), including ones only disabled models use."""
        return self._providers

    async def aclose(self) -> None:
        """Close every provider. One failing close is logged and doesn't stop the others."""
        for name, provider in self._providers.items():
            try:
                await provider.aclose()
            except Exception:
                logger.exception("Failed to close provider %s", name)

    def __repr__(self) -> str:
        return f"ModelRegistry(models={sorted(self._routes)!r}, providers={sorted(self._providers)!r})"


# --- Loading ---


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # json.loads keeps the last of repeated keys by default; a repeated model or provider is
    # almost certainly a mistake, so it's an error instead.
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _describe_validation_errors(exc: ValidationError) -> list[str]:
    # Locations and messages only: input values are left out, so nothing from the file
    # (such as a URL) is echoed back.
    lines = []
    for error in exc.errors(include_url=False, include_input=False, include_context=False):
        location = ".".join(str(part) for part in error["loc"]) or "(top level)"
        lines.append(f"  {location}: {error['msg']}")
    return lines


def parse_registry_config(text: str, source: str = "model registry") -> RegistryConfig:
    """Parse and validate the JSON configuration, or raise RegistryConfigError."""
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKeyError as exc:
        raise RegistryConfigError(
            f"{source}: duplicate key {exc.key!r} (a key may appear only once in each JSON object)"
        ) from None
    except json.JSONDecodeError as exc:
        raise RegistryConfigError(f"{source}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from None
    try:
        return RegistryConfig.model_validate(raw)
    except ValidationError as exc:
        details = "\n".join(_describe_validation_errors(exc))
        raise RegistryConfigError(f"{source}: invalid configuration:\n{details}") from None


def build_registry(config: RegistryConfig, environ: Mapping[str, str]) -> ModelRegistry:
    """Construct each provider once and the routes that share them.

    Every API key variable is checked before any provider is built, so a missing one fails
    without leaving HTTP clients behind.
    """
    api_keys: dict[str, str] = {}
    missing = []
    for name, provider_config in config.providers.items():
        variable = getattr(provider_config, "api_key_env", None)
        if variable is None:
            continue
        value = environ.get(variable)
        if value is None or not value.strip():
            missing.append(f"  providers.{name}.api_key_env: environment variable {variable} is not set or empty")
        else:
            api_keys[name] = value.strip()
    if missing:
        raise RegistryConfigError("model registry: missing API keys:\n" + "\n".join(missing))

    providers = {
        name: create_provider(name, provider_config, api_key=api_keys.get(name))
        for name, provider_config in config.providers.items()
    }
    routes = {
        public_id: ModelRoute(public_id, providers[model.provider], model.provider_model_id)
        for public_id, model in config.models.items()
    }
    enabled = [routes[public_id] for public_id, model in config.models.items() if model.enabled]
    disabled = [routes[public_id] for public_id, model in config.models.items() if not model.enabled]
    return ModelRegistry(providers, enabled, disabled)


def load_registry(path: Path, environ: Mapping[str, str]) -> ModelRegistry:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else type(exc).__name__
        raise RegistryConfigError(f"cannot read model registry file {path}: {reason}") from None
    return build_registry(parse_registry_config(text, source=str(path)), environ)


def default_registry() -> ModelRegistry:
    """Used when no registry file is configured: only the fake provider and its FAKE_MODEL."""
    config = RegistryConfig(
        providers={"fake": FakeProviderConfig(type="fake")},
        models={FAKE_MODEL: ModelConfig(provider="fake", provider_model_id=FAKE_MODEL)},
    )
    return build_registry(config, environ={})


LEGACY_PROVIDER_MESSAGE = (
    "CHAT_PROVIDER={value!r} is no longer supported: providers and models are now configured in "
    "a model registry file. Set MODEL_REGISTRY_FILE (a relative path is resolved against the "
    "backend directory) and describe the provider there; see backend/models.example.json and "
    "the README section 'Model registry' for the migration from OPENAI_COMPATIBLE_*. To keep "
    "using only the fake provider, remove CHAT_PROVIDER or set it to 'fake'."
)


def _process_environment() -> dict[str, str]:
    """os.environ over backend/.env: the same precedence pydantic-settings uses for Settings,
    so API key variables may live in .env like every other setting."""
    from_file = {key: value for key, value in dotenv_values(ENV_FILE).items() if value is not None}
    return {**from_file, **os.environ}


def registry_from_settings(settings: Settings, environ: Mapping[str, str] | None = None) -> ModelRegistry:
    """The registry the application runs with. Raises RegistryConfigError on any problem."""
    if settings.model_registry_file is None:
        if settings.chat_provider != "fake":
            raise RegistryConfigError(LEGACY_PROVIDER_MESSAGE.format(value=settings.chat_provider))
        logger.info("No MODEL_REGISTRY_FILE configured; serving only %s from the fake provider", FAKE_MODEL)
        return default_registry()
    if settings.chat_provider != "fake":
        logger.warning("CHAT_PROVIDER is ignored because MODEL_REGISTRY_FILE is set; remove it")
    registry = load_registry(settings.model_registry_file, _process_environment() if environ is None else environ)
    logger.info("Loaded %r from %s", registry, settings.model_registry_file)
    return registry
