from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.tokens import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_refresh_token,
    new_refresh_token,
)

SECRET = "unit-test-secret-" + "x" * 32
USER_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
TTL = timedelta(minutes=15)


def _token(**overrides) -> str:  # noqa: ANN003
    return create_access_token(USER_ID, SESSION_ID, **{"secret": SECRET, "ttl": TTL, **overrides})


def _claims(**overrides) -> dict:  # noqa: ANN003
    now = int(datetime.now(UTC).timestamp())
    base = {
        "sub": str(USER_ID),
        "sid": str(SESSION_ID),
        "typ": "access",
        "jti": "abc",
        "iat": now,
        "exp": now + 900,
    }
    return {**base, **overrides}


def test_round_trip() -> None:
    claims = decode_access_token(_token(), secret=SECRET)
    assert claims.user_id == USER_ID
    assert claims.session_id == SESSION_ID


def test_every_token_is_unique() -> None:
    assert _token() != _token()


def test_expired_token_is_rejected() -> None:
    old = _token(now=datetime.now(UTC) - timedelta(hours=1))
    with pytest.raises(InvalidTokenError):
        decode_access_token(old, secret=SECRET)


def test_expired_token_can_still_identify_its_session_for_logout() -> None:
    old = _token(now=datetime.now(UTC) - timedelta(hours=1))
    claims = decode_access_token(old, secret=SECRET, verify_expiry=False)
    assert claims.session_id == SESSION_ID


def test_wrong_secret_is_rejected() -> None:
    with pytest.raises(InvalidTokenError):
        decode_access_token(_token(), secret="another-secret-" + "y" * 32)


def test_tampered_signature_is_rejected() -> None:
    token = _token()
    head, signature = token.rsplit(".", 1)
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(InvalidTokenError):
        decode_access_token(f"{head}.{flipped}", secret=SECRET)


def test_unsigned_alg_none_token_is_rejected() -> None:
    unsigned = jwt.encode(_claims(), key="", algorithm="none")
    with pytest.raises(InvalidTokenError):
        decode_access_token(unsigned, secret=SECRET)


def test_token_of_the_wrong_type_is_rejected() -> None:
    forged = jwt.encode(_claims(typ="refresh"), SECRET, algorithm="HS256")
    with pytest.raises(InvalidTokenError):
        decode_access_token(forged, secret=SECRET)


@pytest.mark.parametrize("missing", ["exp", "iat", "sub", "sid", "typ", "jti"])
def test_a_token_missing_any_required_claim_is_rejected(missing: str) -> None:
    claims = _claims()
    del claims[missing]
    with pytest.raises(InvalidTokenError):
        decode_access_token(jwt.encode(claims, SECRET, algorithm="HS256"), secret=SECRET)


def test_non_uuid_subject_is_rejected() -> None:
    forged = jwt.encode(_claims(sub="not-a-uuid"), SECRET, algorithm="HS256")
    with pytest.raises(InvalidTokenError):
        decode_access_token(forged, secret=SECRET)


@pytest.mark.parametrize("garbage", ["", "abc", "a.b.c", "....", "Bearer xyz"])
def test_garbage_is_rejected(garbage: str) -> None:
    with pytest.raises(InvalidTokenError):
        decode_access_token(garbage, secret=SECRET)


def test_refresh_tokens_are_random_and_only_the_digest_is_derivable() -> None:
    first, first_digest = new_refresh_token()
    second, second_digest = new_refresh_token()
    assert first != second and first_digest != second_digest
    assert len(first) >= 60  # 48 random bytes, url-safe encoded
    assert hash_refresh_token(first) == first_digest
    assert len(first_digest) == 64 and first not in first_digest
