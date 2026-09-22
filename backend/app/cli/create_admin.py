"""Create the first admin, or reset an admin's password:

    docker compose run --rm backend python -m app.cli.create_admin --email you@example.com

The password is read from a hidden prompt (or the ADMIN_PASSWORD environment variable for
automation), never from a command-line argument, so it does not end up in shell history.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Sequence

from app.core.config import get_settings
from app.core.database import session_scope
from app.core.logging_config import configure_logging
from app.core.password_policy import WeakPasswordError
from app.services.users import UserExistsError, create_admin


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an admin user.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--full-name", default=None)
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="if the admin already exists, set a new password and sign out all sessions",
    )
    return parser.parse_args(argv)


def _read_password() -> str | None:
    from_env = os.environ.get("ADMIN_PASSWORD")
    if from_env:
        return from_env
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        print("Passwords do not match.", file=sys.stderr)
        return None
    return first


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    password = _read_password()
    if password is None:
        return 1

    try:
        with session_scope() as session:
            user, outcome = create_admin(
                session,
                email=args.email,
                password=password,
                full_name=args.full_name,
                reset_existing=args.reset_password,
            )
    except WeakPasswordError as exc:
        print("Password rejected:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - password {problem}", file=sys.stderr)
        return 1
    except UserExistsError:
        print(
            "A user with this email already exists. Use --reset-password to change an "
            "existing admin's password.",
            file=sys.stderr,
        )
        return 1

    print(f"Admin {user.email}: {outcome}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
