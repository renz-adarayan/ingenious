"""
Environment variable handling and configuration loading.

This module handles environment variable processing and
provides utilities for loading configuration from different sources.
"""

import os
from typing import TYPE_CHECKING

from pydantic_settings import SettingsConfigDict

if TYPE_CHECKING:
    from .main_settings import IngeniousSettings


def get_settings_config() -> SettingsConfigDict:
    """Get the standard settings configuration for pydantic-settings."""
    return SettingsConfigDict(
        env_prefix="INGENIOUS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )


def load_from_env_file(env_file: str = ".env") -> "IngeniousSettings":
    """Load settings from a specific .env file."""
    from .main_settings import IngeniousSettings

    return IngeniousSettings(_env_file=env_file)


def create_minimal_config() -> "IngeniousSettings":
    """Create a minimal configuration for development."""
    from .main_settings import IngeniousSettings
    from .models import (
        LoggingSettings,
        ModelSettings,
        WebAuthenticationSettings,
        WebSettings,
    )

    return IngeniousSettings(
        models=[
            ModelSettings(
                model="gpt-4o-mini",
                api_type="rest",
                api_version="2023-03-15-preview",
                api_key=os.getenv("AZURE_OPENAI_API_KEY", "test-api-key"),
                base_url=os.getenv(
                    "AZURE_OPENAI_BASE_URL", "https://test.openai.azure.com/"
                ),
                deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            )
        ],
        logging=LoggingSettings(root_log_level="debug", log_level="debug"),
        web_configuration=WebSettings(
            # nosec B104: binding to all interfaces needed for containerized deployment
            ip_address="0.0.0.0",
            port=8000,
            type="fastapi",
            asynchronous=False,
            authentication=WebAuthenticationSettings(
                enable=False, username="admin", password="", type="basic"
            ),
        ),
    )
