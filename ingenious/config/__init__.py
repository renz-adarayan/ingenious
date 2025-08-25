"""
Configuration module for Ingenious.

This module provides the public API for configuration management,
maintaining backward compatibility while exposing a clean interface.

Public API:
    - IngeniousSettings: Main configuration class
    - get_config(): Get configuration instance
    - Various configuration model classes for type hints
"""

from .auth_config import AzureAuthConfig
from .environment import create_minimal_config, load_from_env_file
from .main_settings import IngeniousSettings
from .models import (
    AzureSearchSettings,
    AzureSqlSettings,
    ChatHistorySettings,
    ChatServiceSettings,
    FileStorageContainerSettings,
    FileStorageSettings,
    LocalSqlSettings,
    LoggingSettings,
    ModelSettings,
    ReceiverSettings,
    ToolServiceSettings,
    WebAuthenticationSettings,
    WebSettings,
)

__all__ = [
    "AzureAuthConfig",
    # Main settings class
    "IngeniousSettings",
    # Factory functions
    "get_config",
    "load_from_env_file",
    "create_minimal_config",
    # Configuration models
    "ChatHistorySettings",
    "ModelSettings",
    "ChatServiceSettings",
    "ToolServiceSettings",
    "LoggingSettings",
    "AzureSearchSettings",
    "AzureSqlSettings",
    "WebAuthenticationSettings",
    "WebSettings",
    "LocalSqlSettings",
    "FileStorageContainerSettings",
    "FileStorageSettings",
    "ReceiverSettings",
]


def get_config(project_path: str = "") -> IngeniousSettings:
    """
    Get configuration using pydantic-settings system.

    This function provides configuration management that:
    - Automatically loads environment variables
    - Supports .env files
    - Provides validation with helpful error messages
    - Uses nested configuration models

    Args:
        project_path: Optional project path (for backward compatibility)

    Returns:
        IngeniousSettings: The loaded and validated configuration
    """
    from ingenious.core.structured_logging import get_logger

    logger = get_logger(__name__)

    try:
        settings = IngeniousSettings()
        settings.validate_configuration()
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
