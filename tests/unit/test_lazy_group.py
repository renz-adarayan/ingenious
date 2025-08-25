"""
Tests for ingenious.utils.lazy_group module
"""

from unittest.mock import Mock, patch

import pytest
from click import Command, Context
from typer.core import TyperGroup

from ingenious.utils.lazy_group import LazyGroup


class TestLazyGroup:
    """Test cases for LazyGroup class"""

    def test_init(self):
        """Test LazyGroup initialization"""
        group = LazyGroup()
        assert isinstance(group, TyperGroup)
        assert hasattr(group, "_loaders")
        # Document processing commands have been moved to ingenious-aux
        assert isinstance(group._loaders, dict)

    def test_list_commands_basic(self):
        """Test list_commands returns sorted command names"""
        group = LazyGroup()
        ctx = Mock(spec=Context)

        # Mock the parent method to return some commands
        with patch.object(TyperGroup, "list_commands", return_value=["cmd1", "cmd2"]):
            commands = group.list_commands(ctx)

            # Should include parent commands (document processing moved to ingenious-aux)
            assert "cmd1" in commands
            assert "cmd2" in commands
            # Should be sorted
            assert commands == sorted(commands)

    def test_list_commands_deduplication(self):
        """Test list_commands removes duplicates"""
        group = LazyGroup()
        ctx = Mock(spec=Context)

        # Mock parent to return commands
        with patch.object(
            TyperGroup, "list_commands", return_value=["other-cmd", "test-cmd"]
        ):
            commands = group.list_commands(ctx)

            # Should not have duplicates
            assert "other-cmd" in commands
            assert "test-cmd" in commands
            # Document processing commands moved to ingenious-aux
            assert len(commands) == len(set(commands))  # No duplicates

    def test_get_command_main_command_exists(self):
        """Test get_command returns main command when it exists"""
        group = LazyGroup()
        ctx = Mock(spec=Context)
        mock_command = Mock(spec=Command)

        with patch.object(TyperGroup, "get_command", return_value=mock_command):
            result = group.get_command(ctx, "some-main-command")
            assert result is mock_command

    def test_get_command_unknown_command(self):
        """Test get_command returns None for unknown commands"""
        group = LazyGroup()
        ctx = Mock(spec=Context)

        with patch.object(TyperGroup, "get_command", return_value=None):
            result = group.get_command(ctx, "unknown-command")
            assert result is None

    @pytest.mark.skip(reason="Document processing commands moved to ingenious-aux")
    def test_get_command_lazy_load_success(self):
        """Test successful lazy loading of a command - no longer applicable"""
        pass

    @pytest.mark.skip(reason="Document processing commands moved to ingenious-aux")
    def test_get_command_lazy_load_already_click_command(self):
        """Test lazy loading when sub_app is already a Click command - no longer applicable"""
        pass

    @pytest.mark.skip(reason="Document processing commands moved to ingenious-aux")
    def test_get_command_lazy_load_module_not_found(self):
        """Test lazy loading when module is not found - no longer applicable"""
        pass

    def test_loaders_registry_structure(self):
        """Test the structure of the _loaders registry"""
        group = LazyGroup()

        # Registry should be empty after moving document processing commands
        assert isinstance(group._loaders, dict)
        assert len(group._loaders) == 0

    def test_class_exports(self):
        """Test that LazyGroup is properly exported"""
        from ingenious.utils.lazy_group import __all__

        assert "LazyGroup" in __all__
