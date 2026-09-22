"""Password hashing with argon2id (the OWASP-recommended default)."""

from __future__ import annotations

from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.core.config import get_settings


@lru_cache
def _hasher() -> PasswordHasher:
    settings = get_settings()
    return PasswordHasher(
        time_cost=settings.argon2_time_cost,
        memory_cost=settings.argon2_memory_cost_kib,
        parallelism=settings.argon2_parallelism,
    )


def hash_password(password: str) -> str:
    return _hasher().hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher().verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """True if the stored hash uses weaker parameters than the current configuration."""
    return _hasher().check_needs_rehash(password_hash)


@lru_cache
def _dummy_hash() -> str:
    return _hasher().hash("dummy-password-used-only-to-equalise-timing")


def verify_dummy(password: str) -> None:
    """Spend the same time as a real verification (used when the account does not exist),
    so response timing does not reveal which emails are registered."""
    verify_password(password, _dummy_hash())
