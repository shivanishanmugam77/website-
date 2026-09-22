from __future__ import annotations

from app.core.passwords import hash_password, needs_rehash, verify_dummy, verify_password


def test_hash_is_argon2id_and_never_contains_the_password() -> None:
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed.startswith("$argon2id$")
    assert "correct-horse" not in hashed


def test_hashes_are_salted() -> None:
    assert hash_password("same-password-twice") != hash_password("same-password-twice")


def test_verification() -> None:
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_garbage_hashes_fail_closed_instead_of_raising() -> None:
    assert verify_password("anything", "not-a-real-hash") is False
    assert verify_password("anything", "") is False


def test_fresh_hashes_do_not_need_rehashing() -> None:
    assert needs_rehash(hash_password("correct-horse-battery-staple")) is False


def test_dummy_verification_runs_without_error() -> None:
    assert verify_dummy("whatever") is None
