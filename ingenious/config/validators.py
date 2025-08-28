"""
Configuration validation logic.

This module contains validation functions and methods
for ensuring configuration integrity.
"""

import os
from typing import TYPE_CHECKING, List

from .models import ModelSettings

if TYPE_CHECKING:
    from .main_settings import IngeniousSettings


def validate_models_not_empty(models: List[ModelSettings]) -> List[ModelSettings]:
    """Ensure at least one model is configured."""
    if not models:
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        base_url = os.getenv("AZURE_OPENAI_BASE_URL", "")

        if not api_key and not base_url:
            raise ValueError(
                "At least one model must be configured. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL environment variables "
                "or provide model configurations."
            )

        return [
            ModelSettings(
                model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini"),
                api_type="rest",
                api_version="2023-03-15-preview",
                api_key=api_key,
                base_url=base_url,
                deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            )
        ]
    return models


def validate_configuration(settings: "IngeniousSettings") -> None:
    """Validate the complete configuration and provide helpful feedback."""
    errors = []

    if not settings.models:
        errors.append(
            "No models configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL."
        )

    for i, model in enumerate(settings.models):
        if model.api_key and "placeholder" in model.api_key.lower():
            errors.append(
                f"Model {i + 1} has placeholder API key. "
                "Set a valid API key in environment variables."
            )

        if model.base_url and "placeholder" in model.base_url.lower():
            errors.append(
                f"Model {i + 1} has placeholder base URL. "
                "Set a valid base URL in environment variables."
            )

    if (
        settings.web_configuration.authentication.enable
        and not settings.web_configuration.authentication.password
    ):
        errors.append(
            "Web authentication is enabled but no password is set. "
            "Set WEB_AUTH_PASSWORD environment variable."
        )

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        error_msg += "\n\nSee documentation for configuration examples."
        raise ValueError(error_msg)
