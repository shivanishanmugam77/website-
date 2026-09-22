from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select

from app.api.cookies import (
    ACCESS_COOKIE,
    REFRESH_COOKIE,
    clear_session_cookies,
    set_session_cookies,
)
from app.api.deps import ClientInfoDep, CurrentUser, DbSession, SettingsDep
from app.core.password_policy import WeakPasswordError
from app.core.tokens import InvalidTokenError, decode_access_token, hash_refresh_token
from app.models import RefreshToken, User
from app.models.enums import AuditAction
from app.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    UserOut,
)
from app.services import auth as auth_service
from app.services.audit import record_audit
from app.services.sessions import (
    InvalidSessionError,
    revoke_family,
    rotate_session,
    start_session,
)
from app.services.users import EmailAlreadyRegisteredError, create_user

router = APIRouter(prefix="/auth", tags=["auth"])

INVALID_CREDENTIALS = "Invalid email or password"


def _weak_password_error(exc: WeakPasswordError, field: str = "password") -> HTTPException:
    """Same shape as FastAPI's own validation errors, so clients handle one format."""
    detail = [
        {"type": "value_error", "loc": ["body", field], "msg": f"Password {problem}"}
        for problem in exc.problems
    ]
    return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    response: Response,
    db: DbSession,
    settings: SettingsDep,
    client: ClientInfoDep,
) -> User:
    """Create a CUSTOMER account and sign in. Admins cannot be created here."""
    try:
        user = create_user(
            db, email=payload.email, password=payload.password, full_name=payload.full_name
        )
    except WeakPasswordError as exc:
        raise _weak_password_error(exc) from None
    except EmailAlreadyRegisteredError:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Email already registered") from None

    record_audit(
        db,
        AuditAction.USER_REGISTERED,
        actor=user,
        entity_type="user",
        entity_id=str(user.id),
        ip_address=client.ip,
    )
    tokens = start_session(db, settings, user, ip=client.ip, user_agent=client.user_agent)
    db.commit()
    set_session_cookies(response, tokens, settings)
    return user


@router.post("/login", response_model=UserOut)
def login(
    payload: LoginRequest,
    response: Response,
    db: DbSession,
    settings: SettingsDep,
    client: ClientInfoDep,
) -> User:
    try:
        user, tokens = auth_service.login(
            db,
            settings,
            payload.email,
            payload.password,
            ip=client.ip,
            user_agent=client.user_agent,
        )
    except auth_service.InvalidCredentialsError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=INVALID_CREDENTIALS) from None
    set_session_cookies(response, tokens, settings)
    return user


@router.post("/refresh", response_model=UserOut)
def refresh(
    request: Request,
    response: Response,
    db: DbSession,
    settings: SettingsDep,
    client: ClientInfoDep,
) -> User:
    """Rotate the refresh token and issue a new access token."""
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if not refresh_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        user, tokens = rotate_session(
            db, settings, refresh_token, ip=client.ip, user_agent=client.user_agent
        )
    except InvalidSessionError:
        # Cookies are deliberately NOT cleared here: two tabs refreshing at once would
        # otherwise sign each other out. A dead cookie is harmless; login replaces it.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated") from None
    set_session_cookies(response, tokens, settings)
    return user


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,  # a 204 has no body: stop FastAPI inferring one from the annotation
    response_model=None,
)
def logout(
    request: Request,
    response: Response,
    db: DbSession,
    settings: SettingsDep,
    client: ClientInfoDep,
) -> None:
    """End the current session everywhere (server-side), then clear the cookies.

    Always succeeds: logging out while already logged out is not an error.
    """
    family_ids = set()
    actor: User | None = None

    access = request.cookies.get(ACCESS_COOKIE)
    if access:
        try:
            claims = decode_access_token(
                access, secret=settings.secret_key.get_secret_value(), verify_expiry=False
            )
            family_ids.add(claims.session_id)
            actor = db.get(User, claims.user_id)
        except InvalidTokenError:
            pass

    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if refresh_token:
        token = db.scalar(
            select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token))
        )
        if token is not None:
            family_ids.add(token.family_id)
            actor = actor or db.get(User, token.user_id)

    for family_id in family_ids:
        revoke_family(db, family_id)
    if actor is not None:
        record_audit(
            db,
            AuditAction.LOGOUT,
            actor=actor,
            entity_type="user",
            entity_id=str(actor.id),
            ip_address=client.ip,
        )
    db.commit()
    clear_session_cookies(response, settings)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user


@router.post("/change-password", response_model=UserOut)
def change_password(
    payload: ChangePasswordRequest,
    response: Response,
    user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
    client: ClientInfoDep,
) -> User:
    """Change password, sign out every other device and keep this one signed in."""
    try:
        tokens = auth_service.change_password(
            db,
            settings,
            user,
            current_password=payload.current_password,
            new_password=payload.new_password,
            ip=client.ip,
            user_agent=client.user_agent,
        )
    except auth_service.InvalidCredentialsError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect"
        ) from None
    except WeakPasswordError as exc:
        raise _weak_password_error(exc, "new_password") from None
    set_session_cookies(response, tokens, settings)
    return user
