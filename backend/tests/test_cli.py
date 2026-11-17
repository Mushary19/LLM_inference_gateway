"""API key management CLI (python -m app.cli), against the rolled-back test session."""

import io
from contextlib import nullcontext

import pytest
from sqlalchemy import func, select, text

from app import cli
from app.database.models import ApiKey, ClientApplication
from app.services.api_keys import API_KEY_PATTERN, authenticate, hash_api_key


@pytest.fixture
def run(db_session):
    """Run the CLI with the test session; returns (exit code, stdout, stderr)."""

    def invoke(*argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(list(argv), session_scope=lambda: nullcontext(db_session), out=out, err=err)
        return code, out.getvalue(), err.getvalue()

    return invoke


def plaintext_from(output: str) -> str:
    [key] = [line for line in output.splitlines() if API_KEY_PATTERN.fullmatch(line)]
    return key


def stored_rows(db_session) -> str:
    return "\n".join(db_session.execute(text("SELECT row_to_json(k)::text FROM api_keys k")).scalars())


# --- create-app ---


def test_create_app(run, db_session):
    code, out, err = run("create-app", "mobile-app")

    assert code == 0 and "Created application 'mobile-app'" in out and err == ""
    assert db_session.scalar(select(ClientApplication.name)) == "mobile-app"


@pytest.mark.parametrize("name", ["", "   "])
def test_create_app_rejects_blank_names(run, db_session, name):
    code, _, err = run("create-app", name)

    assert code == 1 and "must not be blank" in err
    assert db_session.scalar(select(func.count()).select_from(ClientApplication)) == 0


def test_create_app_rejects_duplicates(run):
    run("create-app", "mobile-app")

    code, _, err = run("create-app", "mobile-app")

    assert code == 1 and "already exists" in err


# --- create-key ---


def test_create_key_refuses_unknown_application(run, db_session):
    code, out, err = run("create-key", "--app", "nope", "--name", "production")

    assert code == 1 and "Unknown application 'nope'" in err and "create-app" in err
    assert out == ""
    assert db_session.scalar(select(func.count()).select_from(ApiKey)) == 0
    assert db_session.scalar(select(func.count()).select_from(ClientApplication)) == 0


def test_create_key_shows_the_key_once_and_stores_only_its_hash(run, db_session):
    run("create-app", "mobile-app")

    code, out, err = run("create-key", "--app", "mobile-app", "--name", "production")

    assert code == 0 and err == ""
    plaintext = plaintext_from(out)
    assert out.count(plaintext) == 1
    assert "shown only once" in out
    record = db_session.scalar(select(ApiKey))
    assert record.key_hash == hash_api_key(plaintext)
    assert f"Key prefix: {record.key_prefix}" in out
    assert plaintext not in stored_rows(db_session)
    assert authenticate(db_session, plaintext).application.name == "mobile-app"


def test_create_key_rejects_blank_name(run, db_session):
    run("create-app", "mobile-app")

    code, _, err = run("create-key", "--app", "mobile-app", "--name", "  ")

    assert code == 1 and "must not be blank" in err
    assert db_session.scalar(select(func.count()).select_from(ApiKey)) == 0


# --- list-keys ---


def test_list_keys_never_shows_secrets(run):
    run("create-app", "mobile-app")
    run("create-app", "web-app")
    _, created_mobile, _ = run("create-key", "--app", "mobile-app", "--name", "production")
    _, created_web, _ = run("create-key", "--app", "web-app", "--name", "staging")
    secrets = [plaintext_from(created_mobile), plaintext_from(created_web)]

    code, out, _ = run("list-keys")

    assert code == 0
    for secret in secrets:
        assert secret not in out
        assert secret[:11] in out  # the prefix identifies the key
        assert secret[12:] not in out
    assert "mobile-app" in out and "web-app" in out and "active" in out

    _, filtered, _ = run("list-keys", "--app", "web-app")
    assert secrets[1][:11] in filtered and secrets[0][:11] not in filtered


def test_list_keys_empty_and_unknown_application(run):
    assert run("list-keys")[1].strip() == "No API keys."

    code, _, err = run("list-keys", "--app", "nope")
    assert code == 1 and "Unknown application" in err


# --- revoke-key ---


def test_revoke_key(run, db_session):
    run("create-app", "mobile-app")
    _, created, _ = run("create-key", "--app", "mobile-app", "--name", "production")
    plaintext = plaintext_from(created)
    prefix = plaintext[:11]

    code, out, _ = run("revoke-key", prefix)

    assert code == 0 and f"Revoked API key {prefix}" in out and plaintext not in out
    assert authenticate(db_session, plaintext) is None
    record = db_session.scalar(select(ApiKey))
    assert not record.is_active and record.revoked_at is not None
    assert "revoked" in run("list-keys")[1]

    code, out, _ = run("revoke-key", prefix)
    assert code == 0 and "already revoked" in out


def test_revoke_unknown_key(run):
    code, _, err = run("revoke-key", "if_unknown0")

    assert code == 1 and "No API key with prefix" in err


def test_cli_created_key_authenticates_requests_until_revoked(run, client):
    run("create-app", "mobile-app")
    _, created, _ = run("create-key", "--app", "mobile-app", "--name", "production")
    headers = {"Authorization": f"Bearer {plaintext_from(created)}"}

    assert client.get("/v1/models", headers=headers).status_code == 200
    run("revoke-key", plaintext_from(created)[:11])
    assert client.get("/v1/models", headers=headers).status_code == 401


def test_help_lists_the_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert all(command in out for command in ("create-app", "create-key", "list-keys", "revoke-key"))


# --- quota commands ---


def test_create_app_gets_the_default_or_given_token_quota(run, db_session):
    from app.core.config import settings
    from app.database.models import TokenQuota

    _, default_out, _ = run("create-app", "defaulted")
    _, custom_out, _ = run("create-app", "custom", "--token-limit", "5000")

    limits = {
        name: db_session.scalar(select(TokenQuota.limit_tokens).join(ClientApplication).where(ClientApplication.name == name))
        for name in ("defaulted", "custom")
    }
    assert limits == {"defaulted": settings.default_token_quota, "custom": 5000}
    assert f"quota of {settings.default_token_quota} tokens" in default_out
    assert "quota of 5000 tokens" in custom_out
    code, _, err = run("create-app", "negative", "--token-limit", "-1")
    assert code == 1 and "must be >= 0" in err


def test_set_and_show_quota(run):
    run("create-app", "mobile-app", "--token-limit", "100")

    code, out, _ = run("set-quota", "--app", "mobile-app", "--limit", "2500")
    assert code == 0 and "set to 2500 tokens" in out

    code, out, _ = run("show-quota", "--app", "mobile-app")
    assert code == 0
    assert out.strip() == "limit=2500 consumed=0 reserved=0 available=2500"


def test_quota_commands_reject_bad_input(run):
    run("create-app", "mobile-app")

    assert run("set-quota", "--app", "nope", "--limit", "5")[0] == 1
    assert run("set-quota", "--app", "mobile-app", "--limit", "-5")[0] == 1
    assert run("show-quota", "--app", "nope")[0] == 1
    assert run("expire-reservations", "--older-than-minutes", "0")[0] == 1


def test_expire_reservations_settles_only_stale_reservations(run, db_session):
    import uuid as uuid_module
    from datetime import timedelta

    from sqlalchemy import func, update

    from app.database.models import QuotaReservation, ReservationStatus
    from app.services import quota

    run("create-app", "mobile-app", "--token-limit", "1000")
    run("create-key", "--app", "mobile-app", "--name", "production")
    key = db_session.scalar(select(ApiKey))
    stale, fresh = (
        quota.reserve(db_session, application_id=key.application_id, api_key_id=key.id,
                      request_id=uuid_module.uuid4(), tokens=100, provider="fake", model="m")
        for _ in range(2)
    )
    db_session.execute(
        update(QuotaReservation).where(QuotaReservation.id == stale.reservation_id)
        .values(created_at=func.now() - timedelta(minutes=90))
    )
    db_session.commit()

    code, out, _ = run("expire-reservations", "--older-than-minutes", "60")

    assert code == 0
    assert "Expired 1 reservation(s)" in out and f"chatcmpl-{stale.request_id.hex}" in out
    db_session.expire_all()
    assert db_session.get(QuotaReservation, stale.reservation_id).status is ReservationStatus.EXPIRED
    assert db_session.get(QuotaReservation, fresh.reservation_id).status is ReservationStatus.RESERVED
    assert run("show-quota", "--app", "mobile-app")[1].strip() == "limit=1000 consumed=100 reserved=100 available=800"
    assert "Expired 0 reservation(s)" in run("expire-reservations")[1]


def test_set_quota_restores_a_deleted_quota_from_its_reservations(run, db_session):
    """The quota row is deleted while reservations exist; set-quota recreates it."""
    import uuid as uuid_module

    from sqlalchemy import text

    from app.database.models import TokenQuota
    from app.providers.base import Usage
    from app.services import quota

    run("create-app", "mobile-app", "--token-limit", "100000")
    run("create-key", "--app", "mobile-app", "--name", "production")
    key = db_session.scalar(select(ApiKey))

    def reserve(tokens):
        return quota.reserve(db_session, application_id=key.application_id, api_key_id=key.id,
                             request_id=uuid_module.uuid4(), tokens=tokens, provider="fake", model="m")

    settled = reserve(500)
    quota.settle(db_session, settled.reservation_id, usage=Usage(40, 60), succeeded=True, may_have_consumed_tokens=True)
    outstanding = reserve(2000)
    db_session.execute(text("DELETE FROM token_quotas WHERE application_id = :id"), {"id": key.application_id})
    db_session.commit()

    code, out, _ = run("set-quota", "--app", "mobile-app", "--limit", "100000")

    assert code == 0
    # Rebuilt from the reservation history, not reset to zero.
    assert "limit=100000 consumed=100 reserved=2000 available=97900" in out
    quota.settle(db_session, outstanding.reservation_id, usage=Usage(300, 200), succeeded=True, may_have_consumed_tokens=True)
    db_session.expire_all()
    restored = db_session.get(TokenQuota, key.application_id)
    assert (restored.consumed_tokens, restored.reserved_tokens) == (600, 0)
    assert quota.quota_inconsistencies(db_session, key.application_id) == []
