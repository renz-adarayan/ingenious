"""
Comprehensive exception hierarchy for Insight Ingenious
======================================================

This module provides a standardized exception hierarchy for all components
of the Ingenious system, with structured error handling, context tracking,
and recovery strategies.

Exception Hierarchy:
- IngeniousError (base)
  ├── ConfigurationError
  │   ├── ConfigFileError
  │   ├── EnvironmentError
  │   └── ValidationError
  ├── DatabaseError
  │   ├── ConnectionError
  │   ├── QueryError
  │   ├── TransactionError
  │   └── MigrationError
  ├── WorkflowError
  │   ├── WorkflowNotFoundError
  │   ├── WorkflowExecutionError
  │   └── WorkflowConfigurationError
  ├── ServiceError
  │   ├── ChatServiceError
  │   ├── AuthenticationError
  │   ├── AuthorizationError
  │   └── ExternalServiceError
  ├── APIError
  │   ├── RequestValidationError
  │   ├── ResponseError
  │   └── RateLimitError
  └── ResourceError
      ├── FileNotFoundError
      ├── PermissionError
      └── StorageError

Features:
- Error codes for programmatic handling
- Rich context information
- Correlation ID support for request tracing
- Recovery strategies and retry mechanisms
- Structured logging integration
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from ingenious.core.structured_logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Error Severity and Categories
# ─────────────────────────────────────────────────────────────────────────────


class ErrorSeverity(Enum):
    """Error severity levels for categorizing impact."""

    LOW = "low"  # Minor issues, system continues normally
    MEDIUM = "medium"  # Moderate issues, some functionality affected
    HIGH = "high"  # Serious issues, major functionality affected
    CRITICAL = "critical"  # System-breaking issues, immediate attention required


class ErrorCategory(Enum):
    """Error categories for classification and handling."""

    CONFIGURATION = "configuration"
    DATABASE = "database"
    WORKFLOW = "workflow"
    SERVICE = "service"
    API = "api"
    RESOURCE = "resource"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    NETWORK = "network"
    PROCESSING = "processing"


# ─────────────────────────────────────────────────────────────────────────────
# Error Context and Metadata
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ErrorContext:
    """Rich context information for errors."""

    # Request tracking
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Operation details
    operation: str = ""
    component: str = ""
    workflow: Optional[str] = None
    service: Optional[str] = None

    # System state
    timestamp: float = field(default_factory=time.time)
    stack_trace: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging and serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None and value != "" and value != {}:
                result[key] = value
        return result

    def add_metadata(self, **kwargs: Any) -> ErrorContext:
        """Add metadata to the context."""
        self.metadata.update(kwargs)
        return self

    def with_stack_trace(self) -> ErrorContext:
        """Capture and add stack trace to context."""
        self.stack_trace = traceback.format_exc()
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Base Exception Class
# ─────────────────────────────────────────────────────────────────────────────


class IngeniousError(Exception):
    """Base exception class for all Ingenious-specific errors.

    This class provides a standardized interface for error handling with:
    - Structured error codes and categories
    - Rich context information with correlation IDs
    - Automatic logging integration
    - Recovery suggestions
    - Error severity classification
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.PROCESSING,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        recovery_suggestion: Optional[str] = None,
        user_message: Optional[str] = None,
    ):
        """Initialize Ingenious error.

        Parameters
        ----------
        message : str
            Technical error description for developers
        error_code : str, optional
            Unique error code for programmatic handling
        category : ErrorCategory
            Error category for classification
        severity : ErrorSeverity
            Error severity level
        context : ErrorContext or dict, optional
            Additional context information
        cause : Exception, optional
            Original exception that triggered this error
        recoverable : bool, default=True
            Whether this error can potentially be recovered from
        recovery_suggestion : str, optional
            Suggestion for how to recover from this error
        user_message : str, optional
            User-friendly error message
        """
        super().__init__(message)

        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.category = category
        self.severity = severity
        self.cause = cause
        self.recoverable = recoverable
        self.recovery_suggestion = recovery_suggestion
        self.user_message = user_message or self._generate_user_message()

        # Handle context
        if context is None:
            self.context = ErrorContext()
        elif isinstance(context, dict):
            self.context = ErrorContext(
                **{
                    k: v
                    for k, v in context.items()
                    if k in ErrorContext.__dataclass_fields__
                }
            )
            self.context.metadata.update(
                {
                    k: v
                    for k, v in context.items()
                    if k not in ErrorContext.__dataclass_fields__
                }
            )
        else:
            self.context = context

        # Set component from class name if not provided
        if not self.context.component:
            self.context.component = self.__class__.__module__

        # Log the error
        self._log_error()

    def _generate_error_code(self) -> str:
        """Generate a default error code based on class name."""
        class_name = self.__class__.__name__
        # Convert CamelCase to UPPER_SNAKE_CASE
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).upper()

    def _generate_user_message(self) -> str:
        """Generate a user-friendly message."""
        return "An error occurred while processing your request. Please try again."

    def _log_error(self) -> None:
        """Log the error with structured context."""
        log_data = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "recoverable": self.recoverable,
            **self.context.to_dict(),
        }

        if self.cause:
            log_data["cause"] = str(self.cause)
            log_data["cause_type"] = self.cause.__class__.__name__

        if self.recovery_suggestion:
            log_data["recovery_suggestion"] = self.recovery_suggestion

        # Log with appropriate level based on severity
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", **log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", **log_data)
        else:
            logger.info("Low severity error occurred", **log_data)

    def with_context(self, **kwargs: Any) -> IngeniousError:
        """Add additional context to the error."""
        self.context.add_metadata(**kwargs)
        return self

    def with_correlation_id(self, correlation_id: str) -> IngeniousError:
        """Set correlation ID for request tracing."""
        self.context.correlation_id = correlation_id
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "user_message": self.user_message,
            "recoverable": self.recoverable,
            "recovery_suggestion": self.recovery_suggestion,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
        }

    def to_json(self) -> str:
        """Convert error to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Errors
# ─────────────────────────────────────────────────────────────────────────────


class ConfigurationError(IngeniousError):
    """Base class for configuration-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("recoverable", False)
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "There is a configuration error. Please check your settings."


class ConfigFileError(ConfigurationError):
    """Raised when configuration file operations fail."""

    def __init__(
        self, message: str, config_path: Optional[str] = None, **kwargs: Any
    ) -> None:
        if config_path:
            kwargs.setdefault("context", {}).update({"config_path": config_path})
        super().__init__(message, **kwargs)


class EnvironmentError(ConfigurationError):
    """Raised when environment variable operations fail."""

    def __init__(
        self, message: str, env_var: Optional[str] = None, **kwargs: Any
    ) -> None:
        if env_var:
            kwargs.setdefault("context", {}).update({"env_var": env_var})
        super().__init__(message, **kwargs)


class ValidationError(ConfigurationError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> None:
        if field:
            kwargs.setdefault("context", {}).update(
                {"field": field, "value": str(value)}
            )
        super().__init__(message, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Database Errors
# ─────────────────────────────────────────────────────────────────────────────


class DatabaseError(IngeniousError):
    """Base class for database-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", ErrorCategory.DATABASE)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("recoverable", True)
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "A database error occurred. Please try again in a moment."


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    def __init__(
        self, message: str, connection_string: Optional[str] = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        if connection_string:
            # Sanitize connection string (remove sensitive info)
            sanitized = self._sanitize_connection_string(connection_string)
            kwargs.setdefault("context", {}).update({"connection_string": sanitized})
        super().__init__(message, **kwargs)

    def _sanitize_connection_string(self, connection_string: str) -> str:
        """Remove sensitive information from connection string."""
        import re

        # Remove passwords and keys
        sanitized = re.sub(
            r"password=([^;]+)", "password=***", connection_string, flags=re.IGNORECASE
        )
        sanitized = re.sub(r"pwd=([^;]+)", "pwd=***", sanitized, flags=re.IGNORECASE)
        return sanitized


class DatabaseQueryError(DatabaseError):
    """Raised when database query execution fails."""

    def __init__(
        self, message: str, query: Optional[str] = None, **kwargs: Any
    ) -> None:
        if query:
            # Truncate long queries
            truncated_query = query[:500] + "..." if len(query) > 500 else query
            kwargs.setdefault("context", {}).update({"query": truncated_query})
        super().__init__(message, **kwargs)


class DatabaseTransactionError(DatabaseError):
    """Raised when database transaction fails."""

    def __init__(
        self, message: str, transaction_id: Optional[str] = None, **kwargs: Any
    ) -> None:
        if transaction_id:
            kwargs.setdefault("context", {}).update({"transaction_id": transaction_id})
        super().__init__(message, **kwargs)


class DatabaseMigrationError(DatabaseError):
    """Raised when database migration fails."""

    def __init__(
        self, message: str, migration_version: Optional[str] = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("recoverable", False)
        if migration_version:
            kwargs.setdefault("context", {}).update(
                {"migration_version": migration_version}
            )
        super().__init__(message, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Errors
# ─────────────────────────────────────────────────────────────────────────────


class WorkflowError(IngeniousError):
    """Base class for workflow-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", ErrorCategory.WORKFLOW)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "A workflow error occurred. Please check your configuration."


class WorkflowNotFoundError(WorkflowError):
    """Raised when a workflow cannot be found."""

    def __init__(
        self, message: str, workflow_name: Optional[str] = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("recoverable", False)
        if workflow_name:
            kwargs.setdefault("context", {}).update({"workflow_name": workflow_name})
        super().__init__(message, **kwargs)


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        step: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if workflow_name:
            kwargs.setdefault("context", {}).update({"workflow_name": workflow_name})
        if step:
            kwargs.setdefault("context", {}).update({"workflow_step": step})
        super().__init__(message, **kwargs)


class WorkflowConfigurationError(WorkflowError):
    """Raised when workflow configuration is invalid."""

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        config_error: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("recoverable", False)
        if workflow_name:
            kwargs.setdefault("context", {}).update({"workflow_name": workflow_name})
        if config_error:
            kwargs.setdefault("context", {}).update({"config_error": config_error})
        super().__init__(message, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Service Errors
# ─────────────────────────────────────────────────────────────────────────────


class ServiceError(IngeniousError):
    """Base class for service-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", ErrorCategory.SERVICE)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "A service error occurred. Please try again."


class ChatServiceError(ServiceError):
    """Raised when chat service operations fail."""

    def __init__(
        self, message: str, service_type: Optional[str] = None, **kwargs: Any
    ) -> None:
        if service_type:
            kwargs.setdefault("context", {}).update({"service_type": service_type})
        super().__init__(message, **kwargs)


class AuthenticationError(ServiceError):
    """Raised when authentication fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", ErrorCategory.AUTHENTICATION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("recoverable", False)
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "Authentication failed. Please check your credentials."


class AuthorizationError(ServiceError):
    """Raised when authorization fails."""

    def __init__(
        self, message: str, required_permission: Optional[str] = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("category", ErrorCategory.AUTHENTICATION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("recoverable", False)
        if required_permission:
            kwargs.setdefault("context", {}).update(
                {"required_permission": required_permission}
            )
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "You don't have permission to perform this action."


class ExternalServiceError(ServiceError):
    """Raised when external service calls fail."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        if service_name:
            kwargs.setdefault("context", {}).update({"service_name": service_name})
        if status_code:
            kwargs.setdefault("context", {}).update({"status_code": status_code})
        super().__init__(message, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# API Errors
# ─────────────────────────────────────────────────────────────────────────────


class APIError(IngeniousError):
    """Base class for API-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", ErrorCategory.API)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "An API error occurred. Please check your request."


class RequestValidationError(APIError):
    """Raised when API request validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("severity", ErrorSeverity.LOW)
        kwargs.setdefault("recoverable", False)
        if field:
            kwargs.setdefault("context", {}).update(
                {"field": field, "value": str(value)}
            )
        super().__init__(message, **kwargs)


class ResponseError(APIError):
    """Raised when API response generation fails."""

    def __init__(
        self, message: str, response_type: Optional[str] = None, **kwargs: Any
    ) -> None:
        if response_type:
            kwargs.setdefault("context", {}).update({"response_type": response_type})
        super().__init__(message, **kwargs)


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("severity", ErrorSeverity.LOW)
        if limit:
            kwargs.setdefault("context", {}).update({"rate_limit": limit})
        if window:
            kwargs.setdefault("context", {}).update({"time_window": window})
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "Rate limit exceeded. Please try again later."


# ─────────────────────────────────────────────────────────────────────────────
# Resource Errors
# ─────────────────────────────────────────────────────────────────────────────


class ResourceError(IngeniousError):
    """Base class for resource-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("category", ErrorCategory.RESOURCE)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)

    def _generate_user_message(self) -> str:
        return "A resource error occurred. Please try again."


class FileNotFoundError(ResourceError):
    """Raised when a file cannot be found."""

    def __init__(
        self, message: str, file_path: Optional[str] = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("recoverable", False)
        if file_path:
            kwargs.setdefault("context", {}).update({"file_path": file_path})
        super().__init__(message, **kwargs)


class PermissionError(ResourceError):
    """Raised when permission to access a resource is denied."""

    def __init__(
        self, message: str, resource_path: Optional[str] = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("recoverable", False)
        if resource_path:
            kwargs.setdefault("context", {}).update({"resource_path": resource_path})
        super().__init__(message, **kwargs)


class StorageError(ResourceError):
    """Raised when storage operations fail."""

    def __init__(
        self, message: str, storage_type: Optional[str] = None, **kwargs: Any
    ) -> None:
        if storage_type:
            kwargs.setdefault("context", {}).update({"storage_type": storage_type})
        super().__init__(message, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Error Collection and Reporting
# ─────────────────────────────────────────────────────────────────────────────


class ErrorCollector:
    """Collects and manages errors for batch processing and reporting."""

    def __init__(self) -> None:
        self.errors: List[IngeniousError] = []
        self.error_counts: Dict[str, int] = {}

    def add_error(self, error: IngeniousError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)

        error_key = f"{error.__class__.__name__}:{error.error_code}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[IngeniousError]:
        """Get errors filtered by severity."""
        return [error for error in self.errors if error.severity == severity]

    def get_errors_by_category(self, category: ErrorCategory) -> List[IngeniousError]:
        """Get errors filtered by category."""
        return [error for error in self.errors if error.category == category]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected errors."""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "by_severity": {
                severity.value: len(self.get_errors_by_severity(severity))
                for severity in ErrorSeverity
            },
            "by_category": {
                category.value: len(self.get_errors_by_category(category))
                for category in ErrorCategory
            },
            "recoverable_errors": sum(1 for e in self.errors if e.recoverable),
            "non_recoverable_errors": sum(1 for e in self.errors if not e.recoverable),
        }

    def export_to_json(self) -> str:
        """Export error collection to JSON."""
        data = {
            "summary": self.get_summary(),
            "errors": [error.to_dict() for error in self.errors],
        }
        return json.dumps(data, indent=2, default=str)

    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
        self.error_counts.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def create_error(
    error_class: Type[IngeniousError], message: str, **kwargs: Any
) -> IngeniousError:
    """Create an error instance with automatic context capture."""
    context = kwargs.get("context", ErrorContext())
    if isinstance(context, ErrorContext):
        context.with_stack_trace()

    return error_class(message, **kwargs)


def handle_exception(
    exc: Exception, operation: str = "", component: str = "", **context_kwargs: Any
) -> IngeniousError:
    """Convert a generic exception to an IngeniousError with context."""

    # Map common exception types to specific Ingenious errors
    error_mapping = {
        FileNotFoundError: ResourceError,
        PermissionError: ResourceError,
        ConnectionError: DatabaseConnectionError,
        TimeoutError: ExternalServiceError,
        ValueError: ValidationError,
        KeyError: ConfigurationError,
    }

    error_class = error_mapping.get(type(exc), IngeniousError)

    # Separate known ErrorContext fields from additional metadata
    context_fields: Dict[str, Any] = {
        "operation": operation,
        "component": component,
    }

    # Add valid ErrorContext fields if they exist in context_kwargs
    valid_fields = {
        "correlation_id",
        "request_id",
        "user_id",
        "session_id",
        "workflow",
        "service",
        "timestamp",
        "stack_trace",
    }

    metadata: Dict[str, Any] = {}
    for key, value in context_kwargs.items():
        if key in valid_fields:
            context_fields[key] = value
        else:
            metadata[key] = value

    if metadata:
        context_fields["metadata"] = metadata

    context = ErrorContext(**context_fields).with_stack_trace()

    return error_class(message=str(exc), cause=exc, context=context)  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    # Context and utilities
    "ErrorContext",
    "ErrorCollector",
    # Base error
    "IngeniousError",
    # Configuration errors
    "ConfigurationError",
    "ConfigFileError",
    "EnvironmentError",
    "ValidationError",
    # Database errors
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseTransactionError",
    "DatabaseMigrationError",
    # Workflow errors
    "WorkflowError",
    "WorkflowNotFoundError",
    "WorkflowExecutionError",
    "WorkflowConfigurationError",
    # Service errors
    "ServiceError",
    "ChatServiceError",
    "AuthenticationError",
    "AuthorizationError",
    "ExternalServiceError",
    # API errors
    "APIError",
    "RequestValidationError",
    "ResponseError",
    "RateLimitError",
    # Resource errors
    "ResourceError",
    "FileNotFoundError",
    "PermissionError",
    "StorageError",
    # Convenience functions
    "create_error",
    "handle_exception",
]
