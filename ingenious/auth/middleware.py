"""
Authentication middleware for FastAPI applications.

This module provides global authentication middleware that can be
optionally enabled to protect all API endpoints.
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ingenious.config.main_settings import IngeniousSettings
from ingenious.core.structured_logging import get_logger
from ingenious.services.auth_dependencies import get_auth_user

logger = get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Global authentication middleware that protects all endpoints.

    This middleware ensures that all API endpoints require authentication
    when authentication is enabled in the configuration.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: IngeniousSettings,
        exempt_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self.config = config
        # Endpoints that should be exempt from authentication
        self.exempt_paths = exempt_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
            "/api/v1/auth/login",  # Allow login endpoint
            "/api/v1/auth/refresh",  # Allow refresh endpoint
            "/api/v1/health",  # Allow health check
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Skip authentication if it's disabled
        if not self.config.web_configuration.authentication.enable:
            return await call_next(request)

        try:
            # Validate authentication
            username = get_auth_user(request, self.config)
            logger.debug(
                f"Authenticated user: {username}", extra={"username": username}
            )

            # Add username to request state for downstream use
            request.state.username = username

            response = await call_next(request)
            return response
        except HTTPException as e:
            # Return authentication error
            logger.warning(
                f"Authentication failed for {request.url.path}: {e.detail}",
                extra={"path": request.url.path, "error": e.detail},
            )
            return JSONResponse(
                content={"detail": e.detail},
                status_code=e.status_code,
                headers=e.headers if hasattr(e, "headers") else {},
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during authentication: {str(e)}",
                extra={"path": request.url.path, "error": str(e)},
                exc_info=True,
            )
            return JSONResponse(
                content={"detail": "Authentication error"},
                status_code=500,
            )


def setup_auth_middleware(
    app: FastAPI, config: IngeniousSettings, exempt_paths: Optional[List[str]] = None
) -> None:
    """
    Setup authentication middleware on the FastAPI app.

    Args:
        app: FastAPI application instance
        config: Ingenious configuration settings
        exempt_paths: Optional list of paths to exempt from authentication
    """
    if config.web_configuration.authentication.enable_global_middleware:
        app.add_middleware(
            AuthenticationMiddleware, config=config, exempt_paths=exempt_paths
        )
        logger.info("Global authentication middleware enabled")
    else:
        logger.info(
            "Global authentication middleware disabled - using per-route authentication"
        )
