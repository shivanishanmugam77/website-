from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import UserRole


class UserOut(BaseModel):
    """Public view of a user. Never includes the password hash or lockout state."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str | None
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


# Password fields are capped so a client cannot make us hash megabytes of input.
class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    # Unknown fields (e.g. "role") are ignored, so a caller cannot self-assign a role.
    email: EmailStr  # length limits are enforced by email-validator itself
    password: str = Field(min_length=1, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=1, max_length=128)
