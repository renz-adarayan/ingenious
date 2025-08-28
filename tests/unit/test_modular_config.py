"""
Unit tests for modular configuration system.

Tests that the refactored configuration modules work together correctly
and maintain backward compatibility.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from ingenious.config import (
    IngeniousSettings,
    create_minimal_config,
    get_config,
    load_from_env_file,
)
from ingenious.config.models import (
    LoggingSettings,
    ModelSettings,
    WebSettings,
)
from ingenious.config.validators import (
    validate_configuration,
    validate_models_not_empty,
)


class TestConfigModels:
    """Test configuration model classes."""

    def test_model_settings_validation(self):
        """Test ModelSettings validation."""
        # Valid model settings
        model = ModelSettings(
            model="gpt-4o-mini",
            api_key="test-key",
            base_url="https://test.openai.azure.com/",
        )
        assert model.model == "gpt-4o-mini"
        assert model.api_key == "test-key"
        assert model.base_url == "https://test.openai.azure.com/"

    def test_model_settings_placeholder_validation(self):
        """Test that placeholder values are rejected."""
        with pytest.raises(ValueError, match="API key is required"):
            ModelSettings(
                model="gpt-4o-mini",
                api_key="placeholder-key",
                base_url="https://test.openai.azure.com/",
            )

        with pytest.raises(ValueError, match="Base URL is required"):
            ModelSettings(
                model="gpt-4o-mini",
                api_key="test-key",
                base_url="placeholder-url",
            )

    def test_logging_settings_validation(self):
        """Test LoggingSettings validation."""
        # Valid logging settings
        logging = LoggingSettings(root_log_level="debug", log_level="info")
        assert logging.root_log_level == "debug"
        assert logging.log_level == "info"

        # Invalid log level
        with pytest.raises(ValueError, match="Log level must be one of"):
            LoggingSettings(root_log_level="invalid")

    def test_web_settings_port_validation(self):
        """Test WebSettings port validation."""
        # Valid port
        web = WebSettings(port=8080)
        assert web.port == 8080

        # Invalid port
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            WebSettings(port=0)

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            WebSettings(port=70000)


class TestConfigValidators:
    """Test configuration validator functions."""

    def test_validate_models_not_empty_with_env_vars(self):
        """Test validate_models_not_empty with environment variables."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_BASE_URL": "https://test.openai.azure.com/",
                "AZURE_OPENAI_MODEL": "gpt-4o-mini",
            },
        ):
            models = validate_models_not_empty([])
            assert len(models) == 1
            assert models[0].model == "gpt-4o-mini"
            assert models[0].api_key == "test-key"

    def test_validate_models_not_empty_without_env_vars(self):
        """Test validate_models_not_empty without environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="At least one model must be configured"
            ):
                validate_models_not_empty([])

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        settings = IngeniousSettings(
            models=[
                ModelSettings(
                    model="gpt-4o-mini",
                    api_key="test-key",
                    base_url="https://test.openai.azure.com/",
                )
            ]
        )
        # Should not raise an exception
        validate_configuration(settings)

    def test_validate_configuration_with_placeholders(self):
        """Test configuration validation with placeholder values."""
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_BASE_URL": "https://test.openai.azure.com/",
            },
        ):
            settings = IngeniousSettings()

            # Temporarily modify model to have placeholder values
            original_key = settings.models[0].api_key
            original_url = settings.models[0].base_url

            # Use object.__setattr__ to bypass pydantic validation
            object.__setattr__(settings.models[0], "api_key", "placeholder-key")
            object.__setattr__(settings.models[0], "base_url", "placeholder-url")

            try:
                with pytest.raises(ValueError, match="Configuration validation failed"):
                    validate_configuration(settings)
            finally:
                # Restore original values
                object.__setattr__(settings.models[0], "api_key", original_key)
                object.__setattr__(settings.models[0], "base_url", original_url)


class TestIngeniousSettings:
    """Test the main IngeniousSettings class."""

    def test_default_settings(self):
        """Test default settings creation."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_BASE_URL": "https://test.openai.azure.com/",
            },
        ):
            settings = IngeniousSettings()
            assert len(settings.models) == 1
            assert settings.models[0].model == "gpt-4o-mini"
            assert settings.profile == "default"
            assert settings.web_configuration.port == 80

    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "INGENIOUS_WEB_CONFIGURATION__PORT": "9000",
                "INGENIOUS_PROFILE": "test",
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_BASE_URL": "https://test.openai.azure.com/",
            },
        ):
            settings = IngeniousSettings()
            assert settings.web_configuration.port == 9000
            assert settings.profile == "test"


class TestConfigFactoryFunctions:
    """Test configuration factory functions."""

    def test_get_config(self):
        """Test get_config function."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_BASE_URL": "https://test.openai.azure.com/",
            },
        ):
            config = get_config()
            assert isinstance(config, IngeniousSettings)
            assert len(config.models) >= 1

    def test_create_minimal_config(self):
        """Test create_minimal_config function."""
        config = create_minimal_config()
        assert isinstance(config, IngeniousSettings)
        assert len(config.models) == 1
        assert config.logging.root_log_level == "debug"
        assert config.web_configuration.port == 8000
        assert not config.web_configuration.authentication.enable

    def test_load_from_env_file(self):
        """Test load_from_env_file function."""
        from unittest.mock import patch

        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("INGENIOUS_PROFILE=test_profile\n")
            f.write("INGENIOUS_WEB_CONFIGURATION__PORT=7000\n")
            f.write("AZURE_OPENAI_API_KEY=test-key\n")
            f.write("AZURE_OPENAI_BASE_URL=https://test.openai.azure.com/\n")
            env_file_path = f.name

        try:
            # Also set environment variables for validation
            with patch.dict(
                "os.environ",
                {
                    "AZURE_OPENAI_API_KEY": "test-key",
                    "AZURE_OPENAI_BASE_URL": "https://test.openai.azure.com/",
                },
            ):
                config = load_from_env_file(env_file_path)
                assert config.profile == "test_profile"
                assert config.web_configuration.port == 7000
        finally:
            os.unlink(env_file_path)


class TestConfigIntegration:
    """Integration tests for the complete configuration system."""

    def test_full_config_load_and_validate(self):
        """Test complete configuration loading and validation."""
        with patch.dict(
            os.environ,
            {
                "INGENIOUS_MODELS__0__MODEL": "gpt-4o-mini",
                "INGENIOUS_MODELS__0__API_KEY": "test-key",
                "INGENIOUS_MODELS__0__BASE_URL": "https://test.openai.azure.com/",
                "INGENIOUS_LOGGING__ROOT_LOG_LEVEL": "debug",
                "INGENIOUS_WEB_CONFIGURATION__PORT": "8080",
                "INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__ENABLE": "false",
            },
        ):
            config = get_config()

            # Verify configuration is loaded correctly
            assert len(config.models) == 1
            assert config.models[0].model == "gpt-4o-mini"
            assert config.logging.root_log_level == "debug"
            assert config.web_configuration.port == 8080
            assert not config.web_configuration.authentication.enable

            # Verify validation passes
            config.validate_configuration()

    def test_modular_import_structure(self):
        """Test that all components can be imported from their new locations."""
        # Test model imports
        from ingenious.config.models import LoggingSettings, ModelSettings

        assert ModelSettings is not None
        assert LoggingSettings is not None

        # Test validator imports
        from ingenious.config.validators import validate_configuration

        assert validate_configuration is not None

        # Test environment imports
        from ingenious.config.environment import create_minimal_config

        assert create_minimal_config is not None

        # Test main settings import
        from ingenious.config.main_settings import IngeniousSettings

        assert IngeniousSettings is not None
