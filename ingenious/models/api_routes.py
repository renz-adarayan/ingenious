from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI

from ingenious.core.structured_logging import get_logger

if TYPE_CHECKING:
    from ingenious.config.main_settings import IngeniousSettings


class IApiRoutes(ABC):
    def __init__(self, config: "IngeniousSettings", app: FastAPI):
        self.config = config
        self.logger = get_logger(__name__)
        self.app = app
        # Note: repositories and services should be obtained via dependency injection in route handlers

    @abstractmethod
    def add_custom_routes(self) -> APIRouter:
        """
        Adds custom routes to the FastAPI app instance. Always returns the router instance.
        """
        pass
