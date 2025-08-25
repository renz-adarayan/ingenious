"""
Tests for ingenious.utils.stage_executor module
"""

import asyncio
from typing import Any
from unittest.mock import Mock, patch

import pytest
from rich.console import Console
from rich.progress import Progress, TaskID

from ingenious.utils.log_levels import LogLevel
from ingenious.utils.stage_executor import (
    IActionCallable,
    ProgressConsoleWrapper,
    stage_executor,
)


class MockActionCallable(IActionCallable):
    """Mock implementation of IActionCallable for testing"""

    def __init__(self, delay: float = 0.0, should_fail: bool = False):
        self.delay = delay
        self.should_fail = should_fail
        self.call_count = 0
        self.last_kwargs: dict[str, Any] = {}

    async def __call__(
        self, progress: ProgressConsoleWrapper, task_id: TaskID, **kwargs
    ) -> None:
        self.call_count += 1
        self.last_kwargs = kwargs
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise Exception("Mock action failed")


class TestIActionCallable:
    """Test cases for IActionCallable abstract base class"""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that IActionCallable cannot be instantiated directly"""
        with pytest.raises(TypeError):
            IActionCallable()

    def test_mock_implementation_works(self):
        """Test that MockActionCallable implements the interface correctly"""
        action = MockActionCallable()
        assert isinstance(action, IActionCallable)
        assert callable(action)


class TestProgressConsoleWrapper:
    """Test cases for ProgressConsoleWrapper class"""

    def test_init(self):
        """Test ProgressConsoleWrapper initialization"""
        mock_progress = Mock(spec=Progress)
        log_level = LogLevel.INFO

        wrapper = ProgressConsoleWrapper(mock_progress, log_level)

        assert wrapper.progress is mock_progress
        assert wrapper.log_level == log_level
        assert wrapper._completed_items == 0
        assert wrapper._failed_items == 0

    def test_completed_items_property(self):
        """Test completed_items property getter and setter"""
        mock_progress = Mock(spec=Progress)
        wrapper = ProgressConsoleWrapper(mock_progress, LogLevel.INFO)

        # Test getter
        assert wrapper.completed_items == 0

        # Test setter
        wrapper.completed_items = 5
        assert wrapper.completed_items == 5
        assert wrapper._completed_items == 5

    def test_failed_items_property(self):
        """Test failed_items property getter and setter"""
        mock_progress = Mock(spec=Progress)
        wrapper = ProgressConsoleWrapper(mock_progress, LogLevel.INFO)

        # Test getter
        assert wrapper.failed_items == 0

        # Test setter
        wrapper.failed_items = 3
        assert wrapper.failed_items == 3
        assert wrapper._failed_items == 3

    def test_print_with_sufficient_log_level(self):
        """Test print method when message level meets threshold"""
        mock_progress = Mock(spec=Progress)
        mock_console = Mock()
        mock_progress.console = mock_console

        wrapper = ProgressConsoleWrapper(mock_progress, LogLevel.INFO)

        # Test print with INFO level (should print)
        wrapper.print("Test message", LogLevel.INFO)

        mock_console.print.assert_called_once_with("Test message", style="info")

    def test_print_with_insufficient_log_level(self):
        """Test print method when message level is below threshold"""
        mock_progress = Mock(spec=Progress)
        mock_console = Mock()
        mock_progress.console = mock_console

        wrapper = ProgressConsoleWrapper(mock_progress, LogLevel.ERROR)

        # Test print with INFO level when threshold is ERROR (should not print)
        wrapper.print("Test message", LogLevel.INFO)

        mock_console.print.assert_not_called()

    def test_print_with_custom_style(self):
        """Test print method with custom style provided"""
        mock_progress = Mock(spec=Progress)
        mock_console = Mock()
        mock_progress.console = mock_console

        wrapper = ProgressConsoleWrapper(mock_progress, LogLevel.INFO)

        # Test print with custom style
        wrapper.print("Test message", LogLevel.INFO, style="custom")

        mock_console.print.assert_called_once_with("Test message", style="custom")

    def test_print_with_additional_args_kwargs(self):
        """Test print method with additional arguments and kwargs"""
        mock_progress = Mock(spec=Progress)
        mock_console = Mock()
        mock_progress.console = mock_console

        wrapper = ProgressConsoleWrapper(mock_progress, LogLevel.INFO)

        # Test print with additional args and kwargs
        wrapper.print("Test message", LogLevel.INFO, "extra_arg", extra_kwarg="value")

        mock_console.print.assert_called_once_with(
            "Test message", "extra_arg", style="info", extra_kwarg="value"
        )

    def test_getattr_delegation(self):
        """Test that unknown attributes are delegated to progress object"""
        mock_progress = Mock(spec=Progress)
        mock_progress.some_attribute = "test_value"
        mock_progress.some_method = Mock(return_value="method_result")

        wrapper = ProgressConsoleWrapper(mock_progress, LogLevel.INFO)

        # Test attribute access
        assert wrapper.some_attribute == "test_value"

        # Test method access
        result = wrapper.some_method("arg")
        assert result == "method_result"
        mock_progress.some_method.assert_called_once_with("arg")


class TestStageExecutor:
    """Test cases for stage_executor class"""

    def test_init(self):
        """Test stage_executor initialization"""
        mock_console = Mock(spec=Console)
        log_level = LogLevel.DEBUG

        executor = stage_executor(log_level, mock_console)

        assert executor.log_level == log_level
        assert executor.console is mock_console

    @pytest.mark.asyncio
    async def test_perform_stage_with_no_action_callables(self):
        """Test perform_stage with empty action callables list"""
        mock_console = Mock(spec=Console)
        executor = stage_executor(LogLevel.INFO, mock_console)

        with patch("ingenious.utils.stage_executor.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value.__enter__.return_value = mock_progress
            mock_progress.add_task.return_value = TaskID(1)

            with patch("time.sleep"):  # Mock sleep to speed up test
                await executor.perform_stage(
                    option=True, action_callables=[], stage_name="Test Stage"
                )

        # Verify that add_task was called
        mock_progress.add_task.assert_called_once()
        # Verify that update was called multiple times for different stages
        assert mock_progress.update.call_count >= 2

    @pytest.mark.asyncio
    async def test_perform_stage_with_action_callables(self):
        """Test perform_stage with action callables"""
        mock_console = Mock(spec=Console)
        executor = stage_executor(LogLevel.INFO, mock_console)

        # Create mock action callables
        action1 = MockActionCallable()
        action2 = MockActionCallable()

        with patch("ingenious.utils.stage_executor.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value.__enter__.return_value = mock_progress
            task_id = TaskID(1)
            mock_progress.add_task.return_value = task_id

            with patch("time.sleep"):  # Mock sleep to speed up test
                await executor.perform_stage(
                    option=True,
                    action_callables=[action1, action2],
                    stage_name="Test Stage",
                    custom_arg="test_value",
                )

        # Verify that both actions were called
        assert action1.call_count == 1
        assert action2.call_count == 1

        # Verify that custom kwargs were passed
        assert action1.last_kwargs["custom_arg"] == "test_value"
        assert action2.last_kwargs["custom_arg"] == "test_value"

    @pytest.mark.asyncio
    async def test_perform_stage_with_option_false(self):
        """Test perform_stage when option is False (stage skipped)"""
        mock_console = Mock(spec=Console)
        executor = stage_executor(LogLevel.INFO, mock_console)

        action = MockActionCallable()

        with patch("ingenious.utils.stage_executor.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value.__enter__.return_value = mock_progress
            mock_progress.add_task.return_value = TaskID(1)

            with patch("time.sleep"):  # Mock sleep to speed up test
                await executor.perform_stage(
                    option=False, action_callables=[action], stage_name="Skipped Stage"
                )

        # Verify that action was not called
        assert action.call_count == 0

        # Verify that "Skipped" status was used
        update_calls = mock_progress.update.call_args_list
        skipped_call = any("Skipped" in str(call) for call in update_calls)
        assert skipped_call

    @pytest.mark.asyncio
    async def test_perform_stage_default_parameters(self):
        """Test perform_stage with default parameters"""
        mock_console = Mock(spec=Console)
        executor = stage_executor(LogLevel.INFO, mock_console)

        with patch("ingenious.utils.stage_executor.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value.__enter__.return_value = mock_progress
            mock_progress.add_task.return_value = TaskID(1)

            with patch("time.sleep"):  # Mock sleep to speed up test
                await executor.perform_stage()

        # Verify that add_task was called with default stage name
        add_task_call = mock_progress.add_task.call_args
        assert "Stage - No Name Provided" in add_task_call[1]["description"]

    @pytest.mark.asyncio
    async def test_perform_stage_none_action_callables(self):
        """Test perform_stage when action_callables is None"""
        mock_console = Mock(spec=Console)
        executor = stage_executor(LogLevel.INFO, mock_console)

        with patch("ingenious.utils.stage_executor.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value.__enter__.return_value = mock_progress
            mock_progress.add_task.return_value = TaskID(1)

            with patch("time.sleep"):  # Mock sleep to speed up test
                await executor.perform_stage(
                    action_callables=None,  # Explicitly pass None
                    stage_name="Test Stage",
                )

        # Should complete without error
        mock_progress.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_stage_runtime_calculation(self):
        """Test that runtime is correctly calculated and displayed"""
        mock_console = Mock(spec=Console)
        executor = stage_executor(LogLevel.INFO, mock_console)

        with patch("ingenious.utils.stage_executor.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value.__enter__.return_value = mock_progress
            mock_progress.add_task.return_value = TaskID(1)

            with patch(
                "time.time", side_effect=[100.0, 101.5]
            ):  # Mock 1.5 second duration
                with patch("time.sleep"):  # Mock sleep to speed up test
                    await executor.perform_stage(stage_name="Timed Stage")

        # Check that the final update call includes runtime information
        final_update_call = mock_progress.update.call_args_list[-1]
        description = final_update_call[1]["description"]
        assert "Runtime:" in description
        assert "00:00:01.500" in description

    @pytest.mark.asyncio
    async def test_progress_console_wrapper_creation(self):
        """Test that ProgressConsoleWrapper is created correctly during stage execution"""
        mock_console = Mock(spec=Console)
        executor = stage_executor(LogLevel.DEBUG, mock_console)

        # Create a mock action that we can use to inspect the progress wrapper
        received_progress = None

        class InspectorAction(IActionCallable):
            async def __call__(
                self, progress: ProgressConsoleWrapper, task_id: TaskID, **kwargs
            ) -> None:
                nonlocal received_progress
                received_progress = progress

        inspector = InspectorAction()

        with patch("ingenious.utils.stage_executor.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value.__enter__.return_value = mock_progress
            mock_progress.add_task.return_value = TaskID(1)

            with patch("time.sleep"):  # Mock sleep to speed up test
                await executor.perform_stage(
                    action_callables=[inspector], stage_name="Inspector Stage"
                )

        # Verify that the action received a ProgressConsoleWrapper
        assert received_progress is not None
        assert isinstance(received_progress, ProgressConsoleWrapper)
        assert received_progress.log_level == LogLevel.DEBUG
