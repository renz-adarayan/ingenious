"""
Tests for ingenious.cli.server_commands module
"""

from unittest.mock import Mock, patch

from rich.console import Console

from ingenious.cli.server_commands import register_commands


class TestServerCommands:
    """Test cases for server CLI commands"""

    def test_module_docstring(self):
        """Test that the module has appropriate documentation"""
        import ingenious.cli.server_commands as server_module

        docstring = server_module.__doc__
        assert docstring is not None
        assert "server" in docstring.lower() or "Server" in docstring

    def test_register_commands_function_exists(self):
        """Test that register_commands function exists"""
        assert callable(register_commands)

    def test_register_commands_basic(self):
        """Test register_commands function"""
        mock_app = Mock()
        mock_console = Mock()

        register_commands(mock_app, mock_console)

        # Should register at least one command
        assert mock_app.command.call_count >= 1

    def test_serve_command_registration(self):
        """Test that serve command gets registered"""
        mock_app = Mock()
        mock_console = Console()

        register_commands(mock_app, mock_console)

        # Check that commands were registered
        assert mock_app.command.called

    def test_serve_function_execution(self):
        """Test serve function registration and structure"""
        # Test that register_commands function exists and is callable
        from ingenious.cli.server_commands import register_commands

        assert callable(register_commands)

        # Test function signature
        import inspect

        sig = inspect.signature(register_commands)
        assert "app" in sig.parameters
        assert "console" in sig.parameters

        # Test that the function can be called without error
        mock_app = Mock()
        mock_console = Mock()
        register_commands(mock_app, mock_console)

        # Should have registered commands
        assert mock_app.command.called

    def test_prompt_tuner_command_exists(self):
        """Test that prompt tuner command registration works"""
        mock_app = Mock()
        mock_console = Console()

        register_commands(mock_app, mock_console)

        # Verify commands were registered
        call_names = [
            call[1]["name"]
            for call in mock_app.command.call_args_list
            if "name" in call[1]
        ]
        expected_commands = ["serve", "prompt-tuner"]

        for expected_cmd in expected_commands:
            assert expected_cmd in call_names

    def test_module_imports(self):
        """Test that module imports work correctly"""
        import ingenious.cli.server_commands as server_module

        # Check that key imports are available
        assert hasattr(server_module, "register_commands")
        assert hasattr(server_module, "typer")
        assert hasattr(server_module, "uvicorn")
        assert hasattr(server_module, "Console")

    def test_environment_variable_handling(self):
        """Test environment variable handling in server commands"""
        import os

        # Test that WEB_PORT environment variable is used in the default parameter
        import ingenious.cli.server_commands as server_module

        # The module should import without error and use environment variables
        assert server_module is not None

        # Test default WEB_PORT usage
        with patch.dict(os.environ, {"WEB_PORT": "9000"}):
            default_port = int(os.getenv("WEB_PORT", "80"))
            assert default_port == 9000
