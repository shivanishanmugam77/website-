from __future__ import annotations

import pytest

from app.core.password_policy import (
    MAX_LENGTH,
    MIN_LENGTH,
    WeakPasswordError,
    ensure_strong_password,
    password_problems,
)


def test_a_long_varied_password_is_accepted() -> None:
    assert password_problems("correct-horse-battery-staple") == []


def test_minimum_length_boundary() -> None:
    assert password_problems("a1b2c3d4e5f6"[: MIN_LENGTH - 1])  # 11 chars: rejected
    assert password_problems("a1b2c3d4e5f6"[:MIN_LENGTH]) == []  # 12 chars: accepted


def test_maximum_length_boundary() -> None:
    at_limit = ("abcde12345" * 13)[:MAX_LENGTH]
    assert password_problems(at_limit) == []
    assert any("at most" in p for p in password_problems(at_limit + "x"))


@pytest.mark.parametrize("weak", ["aaaaaaaaaaaaaaaa", "abababababababab", "1111222211112222"])
def test_low_variety_passwords_are_rejected(weak: str) -> None:
    assert any("different characters" in p for p in password_problems(weak))


def test_password_containing_the_email_name_is_rejected() -> None:
    assert password_problems("xx-Priya.Sharma-xx-2026", "priya.sharma@example.com")
    assert password_problems("unrelated-passphrase-2026", "priya.sharma@example.com") == []


def test_very_short_email_names_are_not_matched() -> None:
    # "al" is too short to be meaningful; matching it would reject far too many passwords.
    assert password_problems("hello-al-friend-2026", "al@example.com") == []


def test_ensure_strong_password_raises_with_all_problems() -> None:
    with pytest.raises(WeakPasswordError) as excinfo:
        ensure_strong_password("aaa")
    assert len(excinfo.value.problems) == 2  # too short AND too few distinct characters
