"""Command-line management of client applications and API keys.

    python -m app.cli create-app <name> [--token-limit N]
    python -m app.cli create-key --app <name> --name <label>
    python -m app.cli list-keys [--app <name>]
    python -m app.cli revoke-key <key_prefix>
    python -m app.cli set-quota --app <name> --limit N
    python -m app.cli show-quota --app <name>
    python -m app.cli expire-reservations [--older-than-minutes M]
    python -m app.cli check-providers [--timeout SECONDS] [--json]

Uses the application's DATABASE_URL. Plaintext keys are printed once, by create-key, and
are never stored or shown again. check-providers needs no database: it loads the model
registry the server would load and probes each provider once (GET /models, no inference).
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from datetime import timedelta
from contextlib import AbstractContextManager
from typing import TextIO

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import TokenQuota
from app.providers.registry import RegistryConfigError, registry_from_settings
from app.services import api_keys, quota
from app.services.provider_health import DEFAULT_PROBE_TIMEOUT_SECONDS, HealthReport, check_registry

MAX_PROBE_TIMEOUT_SECONDS = 300.0


def _create_app(session: Session, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    name = args.name.strip()
    if not name:
        print("Application name must not be blank.", file=err)
        return 1
    if api_keys.get_application(session, name) is not None:
        print(f"Application {name!r} already exists.", file=err)
        return 1
    if args.token_limit is not None and args.token_limit < 0:
        print("Token limit must be >= 0.", file=err)
        return 1
    application = api_keys.create_application(session, name, token_limit=args.token_limit)
    limit = session.get(TokenQuota, application.id).limit_tokens
    session.commit()
    print(f"Created application {application.name!r} ({application.id}) with a quota of {limit} tokens.", file=out)
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


def _set_quota(session: Session, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.limit < 0:
        print("Limit must be >= 0.", file=err)
        return 1
    application = api_keys.get_application(session, args.app)
    if application is None:
        print(f"Unknown application {args.app!r}.", file=err)
        return 1
    quota_row = session.get(TokenQuota, application.id, with_for_update=True)
    if quota_row is None:
        # Recreated from the reservation history, so in-flight and past requests still add up.
        quota_row = quota.restore_missing_quota(session, application.id, args.limit)
    else:
        quota_row.limit_tokens = args.limit
        quota_row.updated_at = func.now()
    session.flush()
    session.refresh(quota_row)
    session.commit()
    print(f"Quota for {application.name!r} set to {args.limit} tokens.", file=out)
    _print_quota(quota_row, out)
    return 0


def _show_quota(session: Session, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    application = api_keys.get_application(session, args.app)
    if application is None:
        print(f"Unknown application {args.app!r}.", file=err)
        return 1
    quota_row = session.get(TokenQuota, application.id)
    if quota_row is None:
        print(f"Application {application.name!r} has no token quota.", file=err)
        return 1
    _print_quota(quota_row, out)
    return 0


def _print_quota(quota_row: TokenQuota, out: TextIO) -> None:
    print(
        f"limit={quota_row.limit_tokens} consumed={quota_row.consumed_tokens} "
        f"reserved={quota_row.reserved_tokens} available={quota_row.available_tokens}",
        file=out,
    )


def _expire_reservations(session: Session, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.older_than_minutes <= 0:
        print("--older-than-minutes must be > 0.", file=err)
        return 1
    expired = quota.expire_stale_reservations(session, timedelta(minutes=args.older_than_minutes))
    print(f"Expired {len(expired)} reservation(s), charging their reserved tokens.", file=out)
    for request_id in expired:
        print(f"  chatcmpl-{request_id.hex}", file=out)
    return 0


def _probe_timeout(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number of seconds") from None
    if not 0 < seconds <= MAX_PROBE_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(f"must be > 0 and <= {MAX_PROBE_TIMEOUT_SECONDS:g}")
    return seconds


async def _run_checks(timeout: float) -> HealthReport:
    # Built inside the event loop that uses and closes it.
    registry = registry_from_settings(settings)
    try:
        return await check_registry(registry, timeout)
    finally:
        await registry.aclose()  # closes every provider, even if checking failed


def _report_as_json(report: HealthReport, source: str | None) -> str:
    return json.dumps(
        {
            "registry": source,
            "ok": report.ok,
            "providers": [
                {"name": c.name, "status": c.status, "reason": c.reason, "latency_ms": c.latency_ms}
                for c in report.providers
            ],
            "models": [
                {
                    "id": m.public_id,
                    "provider": m.provider,
                    "provider_model_id": m.provider_model_id,
                    "enabled": m.enabled,
                    "availability": m.availability,
                }
                for m in report.models
            ],
            "problems": report.problems,
        },
        indent=2,
    )


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows)]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in [headers, *rows]]


def _report_as_text(report: HealthReport, source: str | None) -> str:
    lines = [
        f"Registry: {source}" if source else "Registry: built-in default (MODEL_REGISTRY_FILE is not set)",
        "(the configuration on disk now; a running server uses what it loaded at startup)",
        "",
    ]
    lines += _table(
        ["PROVIDER", "STATUS", "REASON", "LATENCY"],
        [[c.name, c.status, c.reason or "-", f"{c.latency_ms:.0f} ms"] for c in report.providers],
    )
    lines.append("")
    lines += _table(
        ["MODEL", "PROVIDER", "UPSTREAM MODEL", "ENABLED", "AVAILABILITY"],
        [
            [m.public_id, m.provider, m.provider_model_id, "yes" if m.enabled else "no", m.availability]
            for m in report.models
        ],
    )
    lines += [
        "",
        "healthy = reachable and authenticated; it doesn't guarantee inference succeeds.",
        "not_listed = absent from the provider's model list; requests may still work (aliases, tags).",
        "unknown = the provider couldn't be checked, so nothing is known about its models.",
        "",
    ]
    lines += ["Problems:", *(f"  {problem}" for problem in report.problems)] if report.problems else ["No problems found."]
    return "\n".join(lines)


def _check_providers(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    try:
        report = asyncio.run(_run_checks(args.timeout))
    except RegistryConfigError as exc:
        print(f"Model registry configuration error: {exc}", file=err)
        return 2
    source = str(settings.model_registry_file) if settings.model_registry_file else None
    print(_report_as_json(report, source) if args.json else _report_as_text(report, source), file=out)
    return 0 if report.ok else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Manage InferenceFlow API keys, quotas and provider checks.")
    commands = parser.add_subparsers(dest="command", required=True)

    create_app = commands.add_parser("create-app", help="register a client application")
    create_app.add_argument("name")
    create_app.add_argument("--token-limit", type=int, help="token quota (default: DEFAULT_TOKEN_QUOTA)")
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

    set_quota = commands.add_parser("set-quota", help="set an application's token quota limit")
    set_quota.add_argument("--app", required=True, help="application name")
    set_quota.add_argument("--limit", required=True, type=int, help="total tokens allowed")
    set_quota.set_defaults(handler=_set_quota)

    show_quota = commands.add_parser("show-quota", help="show an application's token quota")
    show_quota.add_argument("--app", required=True, help="application name")
    show_quota.set_defaults(handler=_show_quota)

    expire = commands.add_parser(
        "expire-reservations", help="settle reservations left unsettled (crash, cancellation, DB failure)"
    )
    expire.add_argument(
        "--older-than-minutes", type=float, default=60,
        help="only reservations older than this (keep it well above the provider timeout; default 60)",
    )
    expire.set_defaults(handler=_expire_reservations)

    check = commands.add_parser(
        "check-providers",
        help="check provider health and model availability (one GET /models per provider; no inference)",
    )
    check.add_argument(
        "--timeout", type=_probe_timeout, default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help=f"per-provider probe timeout in seconds, independent of inference timeouts "
        f"(default {DEFAULT_PROBE_TIMEOUT_SECONDS:g})",
    )
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    # Exit status: 0 all healthy and listed, 1 problems found, 2 invalid registry configuration.
    check.set_defaults(handler=_check_providers, needs_session=False)
    return parser


def main(
    argv: Sequence[str] | None = None,
    session_scope: Callable[[], AbstractContextManager[Session]] | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    args = _parser().parse_args(argv)
    if not getattr(args, "needs_session", True):
        return args.handler(args, out, err)  # never opens a database session
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
