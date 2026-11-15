"""Command-line management of client applications and API keys.

    python -m app.cli create-app <name>
    python -m app.cli create-key --app <name> --name <label>
    python -m app.cli list-keys [--app <name>]
    python -m app.cli revoke-key <key_prefix>

Uses the application's DATABASE_URL. Plaintext keys are printed once, by create-key, and
are never stored or shown again.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import TextIO

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.services import api_keys


def _create_app(session: Session, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    name = args.name.strip()
    if not name:
        print("Application name must not be blank.", file=err)
        return 1
    if api_keys.get_application(session, name) is not None:
        print(f"Application {name!r} already exists.", file=err)
        return 1
    application = api_keys.create_application(session, name)
    session.commit()
    print(f"Created application {application.name!r} ({application.id}).", file=out)
    return 0


def _create_key(session: Session, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if not args.name.strip():
        print("Key name must not be blank.", file=err)
        return 1
    application = api_keys.get_application(session, args.app)
    if application is None:
        print(
            f"Unknown application {args.app!r}. Create it first: python -m app.cli create-app {args.app!r}",
            file=err,
        )
        return 1
    api_key, plaintext = api_keys.create_api_key(session, application, args.name)
    session.commit()
    print(f"Created API key {api_key.name!r} for application {application.name!r}.", file=out)
    print(f"Key prefix: {api_key.key_prefix}", file=out)
    print("API key (shown only once; store it securely now, it cannot be recovered):", file=out)
    print(plaintext, file=out)
    return 0


def _list_keys(session: Session, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.app is not None and api_keys.get_application(session, args.app) is None:
        print(f"Unknown application {args.app!r}.", file=err)
        return 1
    keys = api_keys.list_api_keys(session, args.app)
    if not keys:
        print("No API keys.", file=out)
        return 0
    print(f"{'PREFIX':<14} {'APPLICATION':<24} {'NAME':<24} {'STATUS':<8} {'CREATED':<20} REVOKED", file=out)
    for key in keys:
        print(
            f"{key.key_prefix:<14} {key.application.name:<24} {key.name:<24} "
            f"{'active' if key.is_active else 'revoked':<8} "
            f"{key.created_at:%Y-%m-%d %H:%M:%S}  "
            f"{f'{key.revoked_at:%Y-%m-%d %H:%M:%S}' if key.revoked_at else '-'}",
            file=out,
        )
    return 0


def _revoke_key(session: Session, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    api_key, changed = api_keys.revoke_api_key(session, args.key_prefix)
    if api_key is None:
        print(f"No API key with prefix {args.key_prefix!r}.", file=err)
        return 1
    if not changed:
        print(f"API key {api_key.key_prefix} was already revoked.", file=out)
        return 0
    session.commit()
    print(f"Revoked API key {api_key.key_prefix} ({api_key.name!r}).", file=out)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Manage InferenceFlow API keys.")
    commands = parser.add_subparsers(dest="command", required=True)

    create_app = commands.add_parser("create-app", help="register a client application")
    create_app.add_argument("name")
    create_app.set_defaults(handler=_create_app)

    create_key = commands.add_parser("create-key", help="create an API key for an existing application")
    create_key.add_argument("--app", required=True, help="application name")
    create_key.add_argument("--name", required=True, help="label for the key, e.g. 'production'")
    create_key.set_defaults(handler=_create_key)

    list_keys = commands.add_parser("list-keys", help="list API keys (never shows secrets)")
    list_keys.add_argument("--app", help="only keys of this application")
    list_keys.set_defaults(handler=_list_keys)

    revoke_key = commands.add_parser("revoke-key", help="revoke an API key by its prefix")
    revoke_key.add_argument("key_prefix", help="e.g. if_k3F9xQ2m (see list-keys)")
    revoke_key.set_defaults(handler=_revoke_key)
    return parser


def main(
    argv: Sequence[str] | None = None,
    session_scope: Callable[[], AbstractContextManager[Session]] | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    args = _parser().parse_args(argv)
    if session_scope is None:
        from app.database.connection import SessionLocal

        session_scope = SessionLocal
    with session_scope() as session:
        try:
            return args.handler(session, args, out, err)
        except IntegrityError as exc:
            session.rollback()
            print(f"Database rejected the change: {exc.orig}", file=err)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
