"""Access-token (JWT) and refresh-token primitives.

Pure functions: the secret and lifetimes are passed in, nothing is read from settings,
and no database is touched. Session state lives in ``services/sessions.py``.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

ALGORITHM = "HS256"
ACCESS_TOKEN_TYPE = "access"  # noqa: S105  (a token *type label*, not a secret)
CLOCK_LEEWAY_SECONDS = 10
REQUIRED_CLAIMS = ("exp", "iat", "sub", "sid", "typ", "jti")


class InvalidTokenError(Exception):
    """The token is missing, malformed, expired, forged or of the wrong type."""


@dataclass(frozen=True)
class AccessTokenClaims:
    user_id: uuid.UUID
    session_id: uuid.UUID


def create_access_token(
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    *,
    secret: str,
    ttl: timedelta,
    now: datetime | None = None,
) -> str:
    issued_at = now or datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "sid": str(session_id),  # the refresh-token family this access token belongs to
        "typ": ACCESS_TOKEN_TYPE,
        "jti": uuid.uuid4().hex,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + ttl).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(
    token: str, *, secret: str, verify_expiry: bool = True
) -> AccessTokenClaims:
    """Validate signature, algorithm, required claims and type. Any failure -> InvalidTokenError.

    ``verify_expiry=False`` is for logout only (identify the session of an expired token).
    """
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],  # never trust the token's own "alg" header
            options={"require": list(REQUIRED_CLAIMS), "verify_exp": verify_expiry},
            leeway=CLOCK_LEEWAY_SECONDS,
        )
        if payload["typ"] != ACCESS_TOKEN_TYPE:
            raise InvalidTokenError("wrong token type")
        return AccessTokenClaims(
            user_id=uuid.UUID(payload["sub"]), session_id=uuid.UUID(payload["sid"])
        )
    except (jwt.PyJWTError, ValueError, KeyError, TypeError) as exc:
        raise InvalidTokenError("invalid access token") from exc


def hash_refresh_token(token: str) -> str:
    # SHA-256 is appropriate here (unlike for passwords): the token is 384 random bits.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_refresh_token() -> tuple[str, str]:
    """Return ``(token, sha256_hex_digest)``. Store only the digest."""
    token = secrets.token_urlsafe(48)
    return token, hash_refresh_token(token)
