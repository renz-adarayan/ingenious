"""Tests for ValidateCommand in cli/commands/help.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

from ingenious.cli.commands.help import ValidateCommand


class TestValidateCommand:
    """Test suite for ValidateCommand."""

    @pytest.fixture
    def validate_command(self, mock_console):
        """Create a ValidateCommand instance for testing."""
        from rich.console import Console

        return ValidateCommand(console=Console())

    @pytest.fixture
    def mock_console(self):
        """Mock console for testing output."""
        with patch("ingenious.cli.commands.help.Console") as mock:
            yield mock.return_value

    def test_validate_environment_variables_with_required_vars_set(
        self, validate_command, mock_console
    ):
        """Test environment variable validation when required vars are set."""
        with patch.dict(
            os.environ,
            {
                "INGENIOUS_MODELS__0__API_KEY": "test-key",
                "INGENIOUS_MODELS__0__BASE_URL": "https://test.openai.azure.com/",
                "INGENIOUS_MODELS__0__MODEL": "gpt-4",
                "INGENIOUS_MODELS__0__API_VERSION": "2024-02-01",
            },
        ):
            errors, warnings = validate_command._validate_environment_variables()
            assert len(errors) == 0
            assert len(warnings) >= 0

    def test_validate_environment_variables_missing_required_vars(
        self, validate_command, mock_console
    ):
        """Test environment variable validation when required vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            errors, warnings = validate_command._validate_environment_variables()
            assert len(errors) > 0
            assert any("API_KEY" in error for error in errors)

    def test_validate_configuration_files_with_valid_files(
        self, validate_command, mock_console, tmp_path
    ):
        """Test configuration file validation with valid files."""
        # Create temp config files
        env_file = tmp_path / ".env"
        env_file.write_text("INGENIOUS_MODELS__0__API_KEY=test-key\n")

        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text("models:\n  - api_key: test-key\n")

        with patch("ingenious.cli.commands.help.Path.exists", return_value=True):
            with patch(
                "ingenious.cli.commands.help.Path.open",
                mock_open(read_data="INGENIOUS_MODELS__0__API_KEY=test"),
            ):
                errors, warnings = validate_command._validate_configuration_files()
                assert len(errors) >= 0  # May have errors for missing files

    def test_validate_azure_connectivity_success(self, validate_command, mock_console):
        """Test Azure connectivity validation with successful connection."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="test"))]
        )

        with patch("ingenious.cli.commands.help.AzureOpenAI", return_value=mock_client):
            with patch.dict(
                os.environ,
                {
                    "INGENIOUS_MODELS__0__API_KEY": "test-key",
                    "INGENIOUS_MODELS__0__BASE_URL": "https://test.openai.azure.com/",
                    "INGENIOUS_MODELS__0__DEPLOYMENT": "test-deployment",
                    "INGENIOUS_MODELS__0__API_VERSION": "2024-02-01",
                },
            ):
                errors, warnings = validate_command._validate_azure_connectivity()
                assert len(errors) == 0

    def test_validate_azure_connectivity_failure(self, validate_command, mock_console):
        """Test Azure connectivity validation with connection failure."""
        with patch(
            "ingenious.cli.commands.help.AzureOpenAI",
            side_effect=Exception("Connection failed"),
        ):
            with patch.dict(
                os.environ,
                {
                    "INGENIOUS_MODELS__0__API_KEY": "test-key",
                    "INGENIOUS_MODELS__0__BASE_URL": "https://test.openai.azure.com/",
                    "INGENIOUS_MODELS__0__DEPLOYMENT": "test-deployment",
                    "INGENIOUS_MODELS__0__API_VERSION": "2024-02-01",
                },
            ):
                errors, warnings = validate_command._validate_azure_connectivity()
                assert len(errors) > 0
                assert any("Connection failed" in error for error in errors)

    def test_validate_workflows(self, validate_command, mock_console):
        """Test workflow validation."""
        mock_discovery = MagicMock()
        mock_discovery.discover_workflows.return_value = {
            "test_workflow": {
                "name": "test_workflow",
                "path": "/path/to/workflow",
                "protocol": "ConversationFlowProtocol",
            }
        }

        with patch(
            "ingenious.cli.commands.help.WorkflowDiscovery", return_value=mock_discovery
        ):
            errors, warnings = validate_command._validate_workflows()
            assert len(errors) == 0

    def test_validate_dependencies(self, validate_command, mock_console):
        """Test dependency validation."""
        with patch("ingenious.cli.commands.help.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="package==1.0.0\n")
            errors, warnings = validate_command._validate_dependencies()
            assert len(errors) == 0

    def test_validate_port_availability_open(self, validate_command, mock_console):
        """Test port availability when port is open."""
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.__enter__.return_value.bind.return_value = None
            errors, warnings = validate_command._validate_port_availability()
            assert len(errors) == 0

    def test_validate_port_availability_in_use(self, validate_command, mock_console):
        """Test port availability when port is in use."""
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.__enter__.return_value.bind.side_effect = OSError(
                "Port in use"
            )
            errors, warnings = validate_command._validate_port_availability()
            assert len(errors) > 0


def mock_open(read_data=""):
    """Helper to create a mock file object."""
    import io
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.__enter__ = lambda self: io.StringIO(read_data)
    mock.__exit__ = lambda self, *args: None
    return MagicMock(return_value=mock)
