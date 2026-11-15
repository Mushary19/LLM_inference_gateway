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
