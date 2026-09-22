from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.core import health
from app.schemas.health import CheckStatus, LiveResponse, ReadyResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LiveResponse)
def live() -> LiveResponse:
    """Liveness: the process is up. Touches no dependency."""
    return LiveResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
)
def ready(response: Response) -> ReadyResponse:
    """Readiness: database reachable and migrated to head, Redis reachable."""
    results = health.run_readiness_checks()
    all_ok = all(result.ok for result in results)
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        status="ok" if all_ok else "degraded",
        checks={r.name: CheckStatus(ok=r.ok, detail=r.detail) for r in results},
    )
