from __future__ import annotations

import getpass
from contextlib import contextmanager

import pytest
from sqlalchemy import select

from app.cli.create_admin import main
from app.core.config import get_settings
from app.core.passwords import verify_password
from app.models import AuditLog, RefreshToken, User
from app.models.enums import AuditAction, UserRole
from app.services.sessions import is_session_active, start_session
from tests.factories import TEST_PASSWORD, make_user


@pytest.fixture(autouse=True)
def _use_the_test_session(monkeypatch, db_session) -> None:  # noqa: ANN001
    """Run the CLI inside the test transaction instead of committing for real."""

    @contextmanager
    def fake_scope():  # noqa: ANN202
        yield db_session

    monkeypatch.setattr("app.cli.create_admin.session_scope", fake_scope)
    monkeypatch.setenv("ADMIN_PASSWORD", TEST_PASSWORD)


def find(session, email: str) -> User | None:  # noqa: ANN001
    return session.scalar(select(User).where(User.email == email))


def test_creates_an_admin_and_audits_it(db_session, capsys) -> None:  # noqa: ANN001
    assert main(["--email", "Boss@Example.com", "--full-name", "The Boss"]) == 0
    admin = find(db_session, "boss@example.com")
    assert admin.role is UserRole.ADMIN and admin.full_name == "The Boss"
    assert verify_password(TEST_PASSWORD, admin.password_hash)
    assert "created" in capsys.readouterr().out
    entry = db_session.scalar(select(AuditLog).where(AuditLog.action == AuditAction.ADMIN_CREATED))
    assert entry.entity_id == str(admin.id) and entry.details["via"] == "cli"


def test_the_password_is_never_printed(capsys) -> None:  # noqa: ANN001
    main(["--email", "quiet@example.com"])
    captured = capsys.readouterr()
    assert TEST_PASSWORD not in captured.out + captured.err


def test_refuses_to_touch_an_existing_user(db_session, capsys) -> None:  # noqa: ANN001
    customer = make_user(db_session, password=TEST_PASSWORD)
    assert main(["--email", customer.email]) == 1
    assert "already exists" in capsys.readouterr().err
    assert customer.role is UserRole.CUSTOMER  # not silently promoted


def test_reset_password_for_an_existing_admin_signs_everyone_out(monkeypatch, db_session) -> None:  # noqa: ANN001
    admin = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    start_session(db_session, get_settings(), admin, ip=None, user_agent=None)
    family = db_session.scalar(
        select(RefreshToken.family_id).where(RefreshToken.user_id == admin.id)
    )
    assert is_session_active(db_session, family)

    monkeypatch.setenv("ADMIN_PASSWORD", "a-different-passphrase-2026")
    assert main(["--email", admin.email, "--reset-password"]) == 0
    db_session.refresh(admin)
    assert verify_password("a-different-passphrase-2026", admin.password_hash)
    assert not is_session_active(db_session, family)


def test_reset_password_never_promotes_a_customer(db_session, capsys) -> None:  # noqa: ANN001
    customer = make_user(db_session, password=TEST_PASSWORD)
    assert main(["--email", customer.email, "--reset-password"]) == 1
    db_session.refresh(customer)
    assert customer.role is UserRole.CUSTOMER
    assert verify_password(TEST_PASSWORD, customer.password_hash)  # unchanged


def test_weak_passwords_are_rejected_and_nothing_is_created(monkeypatch, db_session, capsys) -> None:  # noqa: ANN001
    monkeypatch.setenv("ADMIN_PASSWORD", "short")
    assert main(["--email", "weak@example.com"]) == 1
    assert "at least 12" in capsys.readouterr().err
    assert find(db_session, "weak@example.com") is None


def test_mismatched_interactive_passwords_are_rejected(monkeypatch, db_session, capsys) -> None:  # noqa: ANN001
    monkeypatch.delenv("ADMIN_PASSWORD")
    answers = iter(["first-passphrase-2026", "second-passphrase-2026"])
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": next(answers))
    assert main(["--email", "mismatch@example.com"]) == 1
    assert "do not match" in capsys.readouterr().err
    assert find(db_session, "mismatch@example.com") is None
