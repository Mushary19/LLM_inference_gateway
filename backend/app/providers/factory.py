from typing import TYPE_CHECKING

from app.providers.base import Provider
from app.providers.fake import FakeProvider
from app.providers.openai_compatible import OpenAICompatibleProvider

if TYPE_CHECKING:  # registry.py imports this module, so only import it for type checking
    from app.providers.registry import ProviderConfig


def create_provider(name: str, config: "ProviderConfig", *, api_key: str | None = None) -> Provider:
    """Build the provider instance `name` from its validated configuration.

    `api_key` is the secret read from the environment variable named in the configuration;
    it is passed straight to the provider and kept nowhere else. The caller must aclose()
    the provider at shutdown.
    """
    if config.type == "fake":
        return FakeProvider(name=name)
    if config.type == "openai_compatible":
        return OpenAICompatibleProvider(
            config.base_url, api_key=api_key, timeout_seconds=config.timeout_seconds, name=name
        )
    raise ValueError(f"unknown provider type: {config.type!r}")
