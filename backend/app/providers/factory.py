from app.providers.base import Provider
from app.providers.fake import FakeProvider


def create_provider(name: str) -> Provider:
    """Build the provider selected in settings. Real providers are added here later."""
    if name == "fake":
        return FakeProvider()
    raise ValueError(f"unknown chat provider: {name!r}")
