from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CheckStatus(BaseModel):
    ok: bool
    detail: str | None = None


class LiveResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, CheckStatus]
