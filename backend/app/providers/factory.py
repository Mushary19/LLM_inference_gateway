from app.providers.base import Provider
from app.providers.fake import FakeProvider
from app.providers.openai_compatible import OpenAICompatibleProvider


def create_provider(
    name: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
) -> Provider:
    """Build the provider selected in settings. The caller must aclose() it at shutdown."""
    if name == "fake":
        return FakeProvider()
    if name == "openai_compatible":
        if not base_url:
            raise ValueError("openai_compatible provider requires a base_url")
        return OpenAICompatibleProvider(base_url, api_key=api_key, timeout_seconds=timeout_seconds)
    raise ValueError(f"unknown chat provider: {name!r}")
