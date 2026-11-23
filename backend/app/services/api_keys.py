"""API key generation, storage and verification.

Key format:  if_<public_id>_<secret>
- "if_<public_id>" is the key prefix: 8 random alphanumeric characters, not secret. It is
  stored so keys can be listed and revoked without revealing them.
- <secret> is secrets.token_urlsafe(32): 256 bits of randomness.

Only the SHA-256 hash of the complete key is stored. Plain SHA-256 (no salt or slow hash)
is appropriate because the secret is high-entropy random data, not a user-chosen password.
The plaintext key exists only in the return value of create_api_key().
"""

import hashlib
import re
import secrets
import string
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.database.models import ApiKey, ClientApplication, TokenQuota

KEY_PREFIX = "if_"
PUBLIC_ID_LENGTH = 8
SECRET_BYTES = 32
_PUBLIC_ID_ALPHABET = string.ascii_letters + string.digits
# token_urlsafe(32) is always 43 characters from the URL-safe base64 alphabet.
API_KEY_PATTERN = re.compile(r"^if_[A-Za-z0-9]{8}_[A-Za-z0-9_-]{43}$")
_PREFIX_CONSTRAINT = "uq_api_keys_key_prefix"
MAX_GENERATION_ATTEMPTS = 5


@dataclass(frozen=True)
class GeneratedKey:
    plaintext: str
    prefix: str


def generate_api_key() -> GeneratedKey:
    public_id = "".join(secrets.choice(_PUBLIC_ID_ALPHABET) for _ in range(PUBLIC_ID_LENGTH))
    prefix = f"{KEY_PREFIX}{public_id}"
    return GeneratedKey(plaintext=f"{prefix}_{secrets.token_urlsafe(SECRET_BYTES)}", prefix=prefix)


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def looks_like_api_key(candidate: str) -> bool:
    """Cheap format check, so malformed credentials never reach the database."""
    return API_KEY_PATTERN.fullmatch(candidate) is not None


def create_application(session: Session, name: str, token_limit: int | None = None) -> ClientApplication:
    """Create an application together with its token quota (DEFAULT_TOKEN_QUOTA if not given)."""
    if token_limit is not None and token_limit < 0:
        raise ValueError("token_limit must be >= 0")
    application = ClientApplication(name=name.strip())
    session.add(application)
    session.flush()
    session.add(
        TokenQuota(
            application_id=application.id,
            limit_tokens=settings.default_token_quota if token_limit is None else token_limit,
        )
    )
    session.flush()
    return application


def get_application(session: Session, name: str) -> ClientApplication | None:
    return session.scalar(select(ClientApplication).where(ClientApplication.name == name.strip()))


def create_api_key(
    session: Session, application: ClientApplication, name: str
) -> tuple[ApiKey, str]:
    """Store a new key's hash and metadata; return the record and the plaintext key.

    The plaintext is not stored anywhere and must be shown to the user exactly once.
    A key prefix that collides with an existing one is regenerated.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = generate_api_key()
        # Set the foreign key, not the relationship: a discarded attempt must not linger in
        # application.api_keys after its savepoint is rolled back.
        api_key = ApiKey(
            application_id=application.id,
            name=name.strip(),
            key_prefix=generated.prefix,
            key_hash=hash_api_key(generated.plaintext),
        )
        try:
            with session.begin_nested():
                session.add(api_key)
                session.flush()
        except IntegrityError as exc:
            if getattr(getattr(exc.orig, "diag", None), "constraint_name", None) != _PREFIX_CONSTRAINT:
                raise
            continue  # prefix taken; the savepoint was rolled back, try a new key
        return api_key, generated.plaintext
    raise RuntimeError(f"could not generate a unique API key prefix in {MAX_GENERATION_ATTEMPTS} attempts")


def authenticate(session: Session, plaintext: str) -> ApiKey | None:
    """Return the active key matching `plaintext` (with its application), or None."""
    if not looks_like_api_key(plaintext):
        return None
    return session.scalar(
        select(ApiKey)
        .options(joinedload(ApiKey.application))
        .where(ApiKey.key_hash == hash_api_key(plaintext), ApiKey.is_active.is_(True))
    )


def list_api_keys(session: Session, application_name: str | None = None) -> list[ApiKey]:
    query = (
        select(ApiKey)
        .join(ApiKey.application)
        .options(joinedload(ApiKey.application))
        .order_by(ClientApplication.name, ApiKey.created_at)
    )
    if application_name is not None:
        query = query.where(ClientApplication.name == application_name.strip())
    return list(session.scalars(query))


def revoke_api_key(session: Session, key_prefix: str) -> tuple[ApiKey | None, bool]:
    """Revoke the key with `key_prefix`. Returns (key or None if unknown, whether it changed)."""
    api_key = session.scalar(select(ApiKey).where(ApiKey.key_prefix == key_prefix.strip()))
    if api_key is None or not api_key.is_active:
        return api_key, False
    api_key.is_active = False
    api_key.revoked_at = func.now()
    session.flush()
    session.refresh(api_key)
    return api_key, True
