from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api import (
    admin,
    auth,
    candidates,
    catalogues,
    health,
    panel_editing,
    product_finds,
    suppliers,
)
from app.api.deps import require_trusted_origin

# The origin check applies to every state-changing request under /api.
api_router = APIRouter(dependencies=[Depends(require_trusted_origin)])
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(admin.router)
api_router.include_router(suppliers.router)
api_router.include_router(catalogues.router)
api_router.include_router(panel_editing.router)
api_router.include_router(product_finds.router)
api_router.include_router(candidates.router)
