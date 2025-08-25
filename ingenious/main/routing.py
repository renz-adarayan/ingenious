"""
Route registration for the FastAPI application.

This module handles the registration of all API routes,
including built-in routes and custom extension routes.
"""

from typing import TYPE_CHECKING

from ingenious.api.routes import auth as auth_route
from ingenious.api.routes import chat as chat_route
from ingenious.api.routes import conversation as conversation_route
from ingenious.api.routes import custom_workflows as custom_workflows_route
from ingenious.api.routes import diagnostic as diagnostic_route
from ingenious.api.routes import message_feedback as message_feedback_route
from ingenious.api.routes import prompts as prompts_route
from ingenious.models.api_routes import IApiRoutes
from ingenious.utils.imports import (
    import_class_with_fallback,
    import_module_with_fallback,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ingenious.config import IngeniousSettings


class RouteManager:
    """Manages route registration for the FastAPI application."""

    @staticmethod
    def register_builtin_routes(app: "FastAPI") -> None:
        """Register built-in API routes."""
        app.include_router(
            auth_route.router, prefix="/api/v1/auth", tags=["Authentication"]
        )
        app.include_router(chat_route.router, prefix="/api/v1", tags=["Chat"])
        app.include_router(
            conversation_route.router, prefix="/api/v1", tags=["Conversations"]
        )
        app.include_router(
            diagnostic_route.router, prefix="/api/v1", tags=["Diagnostic"]
        )
        app.include_router(prompts_route.router, prefix="/api/v1", tags=["Prompts"])
        app.include_router(
            message_feedback_route.router, prefix="/api/v1", tags=["Message Feedback"]
        )
        app.include_router(
            custom_workflows_route.router,
            prefix="/api/v1",
            tags=["Custom Workflows"],
        )

    @staticmethod
    def register_custom_routes(app: "FastAPI", config: "IngeniousSettings") -> None:
        """Register custom routes from ingenious extensions."""
        custom_api_routes_module = import_module_with_fallback("api.routes.custom")
        if custom_api_routes_module.__name__ != "ingenious.api.routes.custom":
            custom_api_routes_class = import_class_with_fallback(
                "api.routes.custom", "Api_Routes"
            )
            custom_api_routes_class_instance: IApiRoutes = custom_api_routes_class(
                config, app
            )
            custom_api_routes_class_instance.add_custom_routes()

    @classmethod
    def register_all_routes(cls, app: "FastAPI", config: "IngeniousSettings") -> None:
        """Register all routes (built-in and custom) with the FastAPI app."""
        cls.register_builtin_routes(app)
        cls.register_custom_routes(app, config)
