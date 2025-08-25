"""
Comprehensive tests for the document processing error handling system
===================================================================

This module tests all aspects of the new error handling framework including:
- Error class hierarchy and behavior
- Error codes and context management
- Retry decorators with exponential backoff
- Recovery strategies
- Error reporting utilities
- Integration with logging
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ingenious.errors.processing import (
    # Error codes and context
    ErrorCode,
    ErrorContext,
    # Reporting utilities
    ErrorReporter,
    ExtractionError,
    FallbackEngineStrategy,
    NetworkError,
    # Core error classes
    ProcessingError,
    RetryWithDelayStrategy,
    ValidationError,
    # Convenience functions
    handle_extraction_error,
    handle_network_error,
    handle_validation_error,
    # Retry and recovery
    retry_with_backoff,
)


class TestErrorContext:
    """Test the ErrorContext data class."""

    def test_default_initialization(self):
        """Test creating context with defaults."""
        context = ErrorContext()
        assert context.operation == ""
        assert context.component == ""
        assert context.timestamp > 0
        assert context.retry_count == 0
        assert context.metadata == {}

    def test_initialization_with_values(self):
        """Test creating context with specific values."""
        context = ErrorContext(
            operation="extract",
            component="pymupdf",
            file_path="/test/doc.pdf",
            page_number=5,
        )
        assert context.operation == "extract"
        assert context.component == "pymupdf"
        assert context.file_path == "/test/doc.pdf"
        assert context.page_number == 5

    def test_to_dict_filters_empty_values(self):
        """Test to_dict excludes empty/None values."""
        context = ErrorContext(
            operation="test", component="", file_path=None, page_number=1, metadata={}
        )
        result = context.to_dict()

        assert "operation" in result
        assert "component" not in result  # Empty string filtered
        assert "file_path" not in result  # None filtered
        assert "page_number" in result
        assert "metadata" not in result  # Empty dict filtered

    def test_update_method(self):
        """Test context update functionality."""
        context = ErrorContext(operation="initial")

        updated = context.update(operation="updated", custom_field="custom_value")

        assert updated is context  # Returns self
        assert context.operation == "updated"
        assert context.metadata["custom_field"] == "custom_value"


class TestProcessingError:
    """Test the base ProcessingError class."""

    def test_basic_initialization(self):
        """Test basic error creation."""
        error = ProcessingError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == ErrorCode.UNKNOWN_ERROR
        assert error.recoverable is True
        assert error.cause is None
        assert isinstance(error.context, ErrorContext)

    def test_initialization_with_context_dict(self):
        """Test error creation with context dictionary."""
        error = ProcessingError(
            "Test error",
            error_code=ErrorCode.EXTRACTION_FAILED,
            context={"file_path": "/test/doc.pdf", "page_number": 3},
            recoverable=False,
        )

        assert error.error_code == ErrorCode.EXTRACTION_FAILED
        assert error.context.file_path == "/test/doc.pdf"
        assert error.context.page_number == 3
        assert error.recoverable is False

    def test_initialization_with_context_object(self):
        """Test error creation with ErrorContext object."""
        context = ErrorContext(operation="test", file_path="/test/doc.pdf")
        error = ProcessingError("Test error", context=context)

        assert error.context is context
        assert error.context.operation == "test"

    def test_with_context_method(self):
        """Test adding context after creation."""
        error = ProcessingError("Test error")

        updated = error.with_context(file_path="/test/doc.pdf", page_number=5)

        assert updated is error  # Returns self
        assert error.context.file_path == "/test/doc.pdf"
        assert error.context.page_number == 5

    def test_to_dict_serialization(self):
        """Test error serialization to dictionary."""
        cause = ValueError("Original error")
        error = ProcessingError(
            "Test error",
            error_code=ErrorCode.EXTRACTION_FAILED,
            context={"file_path": "/test/doc.pdf"},
            cause=cause,
            recovery_suggestion="Try again",
        )

        result = error.to_dict()

        assert result["error_type"] == "ProcessingError"
        assert result["error_code"] == "EXTRACTION_FAILED"
        assert result["message"] == "Test error"
        assert result["recoverable"] is True
        assert result["recovery_suggestion"] == "Try again"
        assert result["cause"] == str(cause)
        assert "file_path" in result["context"]

    @patch("ingenious.errors.processing.logger")
    def test_logging_integration(self, mock_logger):
        """Test that errors are logged with structured data."""
        error = ProcessingError(
            "Test error",
            error_code=ErrorCode.EXTRACTION_FAILED,
            context={"file_path": "/test/doc.pdf"},
        )
        assert error is not None

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args

        assert "Processing error occurred" in call_args[0]
        assert call_args[1]["error_type"] == "ProcessingError"
        assert call_args[1]["error_code"] == "EXTRACTION_FAILED"
        assert call_args[1]["message"] == "Test error"


class TestSpecificErrorTypes:
    """Test specific error classes."""

    def test_extraction_error_defaults(self):
        """Test ExtractionError with default values."""
        error = ExtractionError("Extraction failed")

        assert error.error_code == ErrorCode.EXTRACTION_FAILED
        assert error.recoverable is True
        assert "Check logs for specific recovery steps" in error.recovery_suggestion

    def test_extraction_error_recovery_suggestions(self):
        """Test extraction error recovery suggestions."""
        error = ExtractionError(
            "File not found", error_code=ErrorCode.DOCUMENT_NOT_FOUND
        )

        assert "Verify the file path exists" in error.recovery_suggestion

    def test_validation_error_defaults(self):
        """Test ValidationError with default values."""
        error = ValidationError("Validation failed")

        assert error.error_code == ErrorCode.SCHEMA_VALIDATION_FAILED
        assert error.recoverable is False  # Validation errors usually require fixes
        assert "Review and correct the input data format" in error.recovery_suggestion

    def test_network_error_defaults(self):
        """Test NetworkError with default values."""
        error = NetworkError("Network failed")

        assert error.error_code == ErrorCode.NETWORK_CONNECTION_FAILED
        assert error.recoverable is True  # Network errors are often transient
        assert "Check network connectivity and retry" in error.recovery_suggestion

    def test_network_error_recovery_suggestions(self):
        """Test network error recovery suggestions."""
        error = NetworkError("Timeout occurred", error_code=ErrorCode.NETWORK_TIMEOUT)

        assert "Increase timeout or retry" in error.recovery_suggestion


class TestRetryDecorator:
    """Test the retry_with_backoff decorator."""

    def test_successful_execution(self):
        """Test decorator with successful function."""

        @retry_with_backoff(max_retries=3)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_retry_on_recoverable_error(self):
        """Test retry behavior with recoverable errors."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ExtractionError("Temporary failure")
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_recoverable_error(self):
        """Test no retry with non-recoverable errors."""
        call_count = 0

        @retry_with_backoff(max_retries=2)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValidationError("Non-recoverable error")

        with pytest.raises(ValidationError):
            failing_function()

        assert call_count == 1  # Only called once, no retries

    def test_max_retries_exceeded(self):
        """Test behavior when max retries exceeded."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise ExtractionError("Always fails")

        with pytest.raises(ExtractionError) as exc_info:
            always_failing()

        assert call_count == 3  # Initial call + 2 retries
        assert exc_info.value.context.retry_count == 2
        assert exc_info.value.context.max_retries == 2
        assert exc_info.value.context.metadata["final_attempt"] is True

    def test_exponential_backoff_timing(self):
        """Test exponential backoff delay calculation."""
        delays = []
        assert isinstance(delays, list)  # Verify delays is a list for future use

        @retry_with_backoff(max_retries=3, base_delay=0.1, jitter=False)
        def failing_function():
            raise ExtractionError("Test error")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ExtractionError):
                failing_function()

            # Verify exponential backoff: 0.1, 0.2, 0.4
            calls = mock_sleep.call_args_list
            assert len(calls) == 3
            assert abs(calls[0][0][0] - 0.1) < 0.01
            assert abs(calls[1][0][0] - 0.2) < 0.01
            assert abs(calls[2][0][0] - 0.4) < 0.01

    def test_custom_exception_types(self):
        """Test retry with custom exception types."""
        call_count = 0

        @retry_with_backoff(
            max_retries=2, base_delay=0.01, exceptions=(ValueError, RuntimeError)
        )
        def custom_failing():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retriable error")
            return "success"

        result = custom_failing()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_excluded_exception(self):
        """Test no retry on non-specified exception types."""
        call_count = 0

        @retry_with_backoff(max_retries=2, exceptions=(ValueError,))
        def different_error():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Different error")

        with pytest.raises(RuntimeError):
            different_error()

        assert call_count == 1


class TestRecoveryStrategies:
    """Test error recovery strategies."""

    def test_fallback_engine_strategy(self):
        """Test fallback engine recovery strategy."""
        strategy = FallbackEngineStrategy(["pdfminer", "unstructured"])

        # Test can_recover
        extraction_error = ExtractionError(
            "Engine failed",
            error_code=ErrorCode.ENGINE_EXECUTION_FAILED,
            context={"engine_name": "pymupdf"},
        )
        assert strategy.can_recover(extraction_error) is True

        validation_error = ValidationError("Schema error")
        assert strategy.can_recover(validation_error) is False

    def test_fallback_engine_recovery(self):
        """Test fallback engine recovery execution."""
        strategy = FallbackEngineStrategy(["pdfminer", "unstructured"])

        mock_extract = Mock()
        mock_extract.side_effect = [
            ExtractionError("First fallback failed"),
            ["success_result"],
        ]

        error = ExtractionError(
            "Original engine failed",
            error_code=ErrorCode.ENGINE_EXECUTION_FAILED,
            context={"engine_name": "pymupdf"},
        )

        result = strategy.recover(error, mock_extract, "test.pdf")

        assert result == ["success_result"]
        assert mock_extract.call_count == 2
        mock_extract.assert_any_call("test.pdf", engine="pdfminer")
        mock_extract.assert_any_call("test.pdf", engine="unstructured")

    def test_retry_with_delay_strategy(self):
        """Test retry with delay strategy."""
        strategy = RetryWithDelayStrategy(max_retries=2, base_delay=0.01)

        # Test can_recover
        network_error = NetworkError("Timeout", error_code=ErrorCode.NETWORK_TIMEOUT)
        network_error.context.retry_count = 1
        assert strategy.can_recover(network_error) is True

        # Test max retries reached
        network_error.context.retry_count = 2
        assert strategy.can_recover(network_error) is False

    def test_retry_with_delay_recovery(self):
        """Test retry with delay recovery execution."""
        strategy = RetryWithDelayStrategy(max_retries=2, base_delay=0.01)

        mock_operation = Mock(return_value="success")

        error = NetworkError("Temporary failure", error_code=ErrorCode.NETWORK_TIMEOUT)
        error.context.retry_count = 0

        with patch("time.sleep") as mock_sleep:
            result = strategy.recover(error, mock_operation, "arg1", key="value")

        assert result == "success"
        mock_operation.assert_called_once_with("arg1", key="value")
        mock_sleep.assert_called_once()
        assert error.context.retry_count == 1


class TestErrorReporter:
    """Test the ErrorReporter utility."""

    def test_add_error(self):
        """Test adding errors to reporter."""
        reporter = ErrorReporter()

        error1 = ExtractionError("Error 1")
        error2 = NetworkError("Error 2")
        error3 = ExtractionError("Error 3")

        reporter.add_error(error1)
        reporter.add_error(error2)
        reporter.add_error(error3)

        assert len(reporter.errors) == 3
        assert reporter.error_counts["ExtractionError:EXTRACTION_FAILED"] == 2
        assert reporter.error_counts["NetworkError:NETWORK_CONNECTION_FAILED"] == 1

    def test_error_summary(self):
        """Test error summary generation."""
        reporter = ErrorReporter()

        reporter.add_error(ExtractionError("Error 1"))
        reporter.add_error(NetworkError("Error 2"))
        reporter.add_error(ValidationError("Error 3"))

        summary = reporter.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["recoverable_errors"] == 2  # Extraction and Network
        assert summary["non_recoverable_errors"] == 1  # Validation
        assert len(summary["most_common_errors"]) <= 5

    def test_export_to_json(self):
        """Test JSON export functionality."""
        reporter = ErrorReporter()

        error = ExtractionError("Test error", context={"file_path": "/test/doc.pdf"})
        reporter.add_error(error)

        json_output = reporter.export_to_json()
        data = json.loads(json_output)

        assert "summary" in data
        assert "errors" in data
        assert data["summary"]["total_errors"] == 1
        assert len(data["errors"]) == 1
        assert data["errors"][0]["error_type"] == "ExtractionError"

    def test_clear(self):
        """Test clearing reporter data."""
        reporter = ErrorReporter()

        reporter.add_error(ExtractionError("Error"))
        assert len(reporter.errors) == 1

        reporter.clear()
        assert len(reporter.errors) == 0
        assert len(reporter.error_counts) == 0


class TestConvenienceFunctions:
    """Test convenience functions for creating errors."""

    def test_handle_extraction_error(self):
        """Test handle_extraction_error function."""
        error = handle_extraction_error(
            operation="extract_text",
            src="/test/doc.pdf",
            engine="pymupdf",
            page_number=5,
        )

        assert isinstance(error, ExtractionError)
        assert error.message == "Failed to extract_text"
        assert error.context.operation == "extract_text"
        assert error.context.component == "document_processing.extractor"
        assert error.context.file_path == "/test/doc.pdf"
        assert error.context.engine_name == "pymupdf"
        assert error.context.page_number == 5

    def test_handle_network_error(self):
        """Test handle_network_error function."""
        error = handle_network_error(
            url="https://example.com/doc.pdf", operation="download", status_code=404
        )

        assert isinstance(error, NetworkError)
        assert (
            error.message == "Network download failed for https://example.com/doc.pdf"
        )
        assert error.context.operation == "download"
        assert error.context.component == "document_processing.network"
        assert error.context.url == "https://example.com/doc.pdf"
        assert error.context.status_code == 404
        assert error.error_code == ErrorCode.HTTP_ERROR

    def test_handle_validation_error(self):
        """Test handle_validation_error function."""
        error = handle_validation_error(
            field_name="page_number", expected_type="int", actual_value="not_a_number"
        )

        assert isinstance(error, ValidationError)
        assert "page_number" in error.message
        assert "expected int" in error.message
        assert "got str" in error.message
        assert error.context.operation == "validation"
        assert error.context.component == "document_processing.validation"
        assert error.error_code == ErrorCode.TYPE_VALIDATION_FAILED


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_document_extraction_with_fallback(self):
        """Test complete document extraction with fallback strategy."""

        # Mock extraction function that fails with primary engine
        def mock_extract(src, engine="pymupdf"):
            if engine == "pymupdf":
                raise ExtractionError(
                    "PyMuPDF failed",
                    error_code=ErrorCode.ENGINE_EXECUTION_FAILED,
                    context={"engine_name": engine},
                )
            elif engine == "pdfminer":
                return [{"page": 1, "text": "success", "type": "NarrativeText"}]
            else:
                raise ExtractionError("Fallback also failed")

        # Apply retry decorator and fallback strategy
        @retry_with_backoff(max_retries=1, base_delay=0.01)
        def extract_with_fallback(src):
            try:
                return mock_extract(src, engine="pymupdf")
            except ExtractionError as exc:
                if exc.error_code == ErrorCode.ENGINE_EXECUTION_FAILED:
                    strategy = FallbackEngineStrategy(["pdfminer"])
                    return strategy.recover(exc, mock_extract, src)
                raise

        result = extract_with_fallback("test.pdf")
        assert result == [{"page": 1, "text": "success", "type": "NarrativeText"}]

    def test_network_download_with_retry(self):
        """Test network download with retry on timeout."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, exceptions=(NetworkError,))
        def download_with_retry(url):
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise NetworkError(
                    "Timeout",
                    error_code=ErrorCode.NETWORK_TIMEOUT,
                    context={"url": url},
                )

            return b"downloaded_content"

        result = download_with_retry("https://example.com/doc.pdf")
        assert result == b"downloaded_content"
        assert call_count == 3

    def test_error_reporting_workflow(self):
        """Test complete error reporting workflow."""
        reporter = ErrorReporter()

        # Simulate various errors during processing
        errors = [
            ExtractionError("Doc 1 failed", context={"file_path": "/doc1.pdf"}),
            NetworkError(
                "Download failed", context={"url": "https://example.com/doc2.pdf"}
            ),
            ExtractionError("Doc 3 failed", context={"file_path": "/doc3.pdf"}),
            ValidationError("Schema error"),
        ]

        for error in errors:
            reporter.add_error(error)

        summary = reporter.get_error_summary()

        assert summary["total_errors"] == 4
        assert summary["recoverable_errors"] == 3
        assert summary["non_recoverable_errors"] == 1

        # Most common should be ExtractionError
        most_common = summary["most_common_errors"][0]
        assert "ExtractionError" in most_common[0]
        assert most_common[1] == 2

        # Test JSON export
        json_output = reporter.export_to_json()
        data = json.loads(json_output)
        assert len(data["errors"]) == 4


# Additional integration tests with mocked external dependencies
class TestExternalIntegration:
    """Test integration with external libraries and systems."""

    @patch("ingenious.errors.processing.logger")
    def test_logging_structured_data(self, mock_logger):
        """Test that structured logging captures all error data."""
        error = ExtractionError(
            "Test extraction error",
            error_code=ErrorCode.DOCUMENT_CORRUPTED,
            context={
                "file_path": "/test/doc.pdf",
                "engine_name": "pymupdf",
                "file_size": 1024,
            },
            cause=ValueError("Original cause"),
        )
        assert error is not None

        # Verify error was logged
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args

        # Check log message
        assert "Processing error occurred" in call_args[0]

        # Check structured data
        log_data = call_args[1]
        assert log_data["error_type"] == "ExtractionError"
        assert log_data["error_code"] == "DOCUMENT_CORRUPTED"
        assert log_data["message"] == "Test extraction error"
        assert log_data["recoverable"] is True
        assert log_data["file_path"] == "/test/doc.pdf"
        assert log_data["engine_name"] == "pymupdf"
        assert log_data["cause"] == "Original cause"
        assert log_data["cause_type"] == "ValueError"

    @pytest.mark.skip(
        reason="document_processing module moved to ingenious-aux/document-preprocessing"
    )
    def test_requests_exception_mapping(self):
        """Test mapping of requests exceptions to NetworkError."""
        # NOTE: This test has been disabled as the document_processing module
        # has been moved to ingenious-aux/document-preprocessing
        pass

    def test_path_like_objects(self):
        """Test error handling with pathlib.Path objects."""
        path = Path("/test/document.pdf")

        error = handle_extraction_error(operation="extract", src=path, engine="pymupdf")

        assert error.context.file_path == str(path)
        assert error.context.engine_name == "pymupdf"
