"""FastAPI application factory. Run with: uvicorn app.main:create_app --factory"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.router import api_router
from app.core.config import get_settings
from app.core.constants import API_PREFIX, REQUEST_ID_HEADER
from app.core.logging_config import configure_logging
from app.core.middleware import RequestContextMiddleware

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        # Interactive docs are a development convenience only.
        docs_url=None if settings.is_production else f"{API_PREFIX}/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else f"{API_PREFIX}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
    )
    # Added last => outermost: every response, including CORS preflights, gets a request id.
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "unhandled_exception",
            extra={"request_id": request_id, "path": request.url.path},
        )
        # Never echo exception text to clients.
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )

    app.include_router(api_router, prefix=API_PREFIX)
    return app
