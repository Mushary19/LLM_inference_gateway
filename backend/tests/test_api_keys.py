"""API key service: generation, hashing, storage, lookup, revocation. Uses the test database."""

import hashlib

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.exc import IntegrityError

from app.database.connection import engine
from app.database.models import ApiKey, ClientApplication
from app.services import api_keys
from app.services.api_keys import (
    API_KEY_PATTERN,
    GeneratedKey,
    authenticate,
    create_api_key,
    create_application,
    generate_api_key,
    hash_api_key,
    looks_like_api_key,
    revoke_api_key,
)


@pytest.fixture
def application(db_session):
    return create_application(db_session, "billing-service")


@pytest.fixture
def count_statements():
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    yield statements
    event.remove(engine, "before_cursor_execute", record)


# --- Generation and hashing ---


def test_generated_key_format():
    key = generate_api_key()

    assert API_KEY_PATTERN.fullmatch(key.plaintext)
    assert key.plaintext.startswith("if_") and key.prefix.startswith("if_")
    assert key.plaintext.startswith(key.prefix + "_")
    assert len(key.prefix) == 11
    # The secret after the prefix is token_urlsafe(32): 43 characters, 256 bits.
    assert len(key.plaintext) - len(key.prefix) - 1 == 43


def test_generated_keys_are_unique():
    keys = [generate_api_key() for _ in range(1000)]

    assert len({k.plaintext for k in keys}) == 1000
    assert len({k.prefix for k in keys}) == 1000
    assert generate_api_key().plaintext != generate_api_key().plaintext


def test_hash_is_sha256_hex_of_the_full_key():
    key = generate_api_key().plaintext

    assert hash_api_key(key) == hashlib.sha256(key.encode()).hexdigest()
    assert hash_api_key(key) == hash_api_key(key)
    assert hash_api_key(key) != hash_api_key(generate_api_key().plaintext)
    assert len(hash_api_key(key)) == 64


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "if_",
        "sk-1234567890abcdef",
        "if_short_secret",
        "IF_abcdefgh_" + "a" * 43,  # prefix is case-sensitive
        "if_abcdefg_" + "a" * 43,  # public id too short
        "if_abcdefgh_" + "a" * 42,  # secret too short
        "if_abcdefgh_" + "a" * 44,  # secret too long
        "if_abcdefgh_" + "a" * 42 + "!",
        " if_abcdefgh_" + "a" * 43,
        "if_abcdefgh_" + "a" * 43 + "\n",
    ],
)
def test_malformed_keys_are_recognized_without_a_lookup(candidate):
    assert not looks_like_api_key(candidate)


# --- Storage ---


def test_only_the_hash_is_stored(db_session, application):
    record, plaintext = create_api_key(db_session, application, "production")
    secret = plaintext.removeprefix(record.key_prefix + "_")

    stored_row = db_session.execute(
        text("SELECT row_to_json(k)::text FROM api_keys k WHERE id = :id"), {"id": record.id}
    ).scalar_one()

    assert record.key_hash == hash_api_key(plaintext)
    assert record.key_prefix == plaintext[:11]
    assert plaintext not in stored_row and secret not in stored_row
    assert record.is_active and record.revoked_at is None and record.created_at is not None


def test_key_belongs_to_its_application(db_session, application):
    record, _ = create_api_key(db_session, application, "production")

    db_session.expire_all()
    stored = db_session.get(ApiKey, record.id)
    assert stored.application.name == "billing-service"
    assert [k.id for k in stored.application.api_keys] == [record.id]


def test_prefix_collision_is_retried(db_session, application, monkeypatch):
    existing, _ = create_api_key(db_session, application, "first")
    fresh = generate_api_key()
    attempts = iter(
        [GeneratedKey(plaintext=f"{existing.key_prefix}_{'x' * 43}", prefix=existing.key_prefix), fresh]
    )
    monkeypatch.setattr(api_keys, "generate_api_key", lambda: next(attempts))

    record, plaintext = create_api_key(db_session, application, "second")

    assert plaintext == fresh.plaintext and record.key_prefix == fresh.prefix
    assert db_session.scalar(select(ApiKey).where(ApiKey.key_prefix == existing.key_prefix)).name == "first"
    assert {k.name for k in db_session.scalars(select(ApiKey))} == {"first", "second"}


def test_persistent_prefix_collisions_give_up(db_session, application, monkeypatch):
    existing, _ = create_api_key(db_session, application, "first")
    collision = GeneratedKey(plaintext=f"{existing.key_prefix}_{'x' * 43}", prefix=existing.key_prefix)
    monkeypatch.setattr(api_keys, "generate_api_key", lambda: collision)

    with pytest.raises(RuntimeError, match="unique API key prefix"):
        create_api_key(db_session, application, "second")
    assert db_session.scalar(select(ApiKey).where(ApiKey.name == "second")) is None


def test_database_enforces_key_invariants(db_session, application):
    record, _ = create_api_key(db_session, application, "production")

    with pytest.raises(IntegrityError, match="ck_api_keys_revocation_consistent"):
        with db_session.begin_nested():
            db_session.execute(text("UPDATE api_keys SET is_active = false WHERE id = :id"), {"id": record.id})
    with pytest.raises(IntegrityError, match="ck_api_keys_key_hash_sha256_hex"):
        with db_session.begin_nested():
            db_session.execute(text("UPDATE api_keys SET key_hash = 'not-a-hash' WHERE id = :id"), {"id": record.id})
    with pytest.raises(IntegrityError, match="uq_client_applications_name"):
        with db_session.begin_nested():
            db_session.add(ClientApplication(name="billing-service"))
            db_session.flush()
    with pytest.raises(IntegrityError, match="ck_client_applications_name_not_blank"):
        with db_session.begin_nested():
            db_session.add(ClientApplication(name="   "))
            db_session.flush()


# --- Authentication and revocation ---


def test_authenticate_returns_the_active_key_with_its_application(db_session, application):
    record, plaintext = create_api_key(db_session, application, "production")

    found = authenticate(db_session, plaintext)

    assert found is not None and found.id == record.id
    assert found.application.name == "billing-service"


def test_authenticate_rejects_unknown_and_revoked_keys(db_session, application):
    record, plaintext = create_api_key(db_session, application, "production")

    assert authenticate(db_session, generate_api_key().plaintext) is None
    revoke_api_key(db_session, record.key_prefix)
    assert authenticate(db_session, plaintext) is None


def test_malformed_credentials_never_query_the_database(db_session, count_statements):
    assert authenticate(db_session, "Bearer nonsense") is None
    assert authenticate(db_session, "x" * 10_000) is None

    assert count_statements == []


def test_revocation_sets_both_fields_and_is_idempotent(db_session, application):
    record, _ = create_api_key(db_session, application, "production")

    revoked, changed = revoke_api_key(db_session, record.key_prefix)
    again, changed_again = revoke_api_key(db_session, record.key_prefix)

    assert changed and revoked.id == record.id
    assert not revoked.is_active and revoked.revoked_at is not None
    assert again.id == record.id and not changed_again
    assert revoke_api_key(db_session, "if_unknown0") == (None, False)
