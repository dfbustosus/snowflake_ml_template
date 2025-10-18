"""Structured logging with context support.

This module provides a structured logging system that supports:
- Correlation IDs for tracking requests across components
- Context injection (pipeline_id, model_name, user, etc.)
- Multiple output formats (JSON, text)
- Thread-safe context management
- Integration with Python's standard logging

Classes:
    StructuredLogger: Logger with context support
    LogContext: Context manager for adding temporary context
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Literal, Optional

from snowflake_ml_template.utils.logging.formatters import ContextFormatter

# Thread-local storage for log context
_log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})


class StructuredLogger:
    """Structured logger with context support.

    This class wraps Python's standard logger and adds support for
    structured logging with context. It automatically includes:
    - Correlation ID for tracking requests
    - Custom context fields (pipeline_id, model_name, etc.)
    - Timestamp and log level
    - Module and function name

    The logger is thread-safe and uses context variables to maintain
    separate context for each thread/async task.

    Attributes:
        name: Logger name
        logger: Underlying Python logger
        correlation_id: Unique ID for this logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>>
        >>> # Simple logging
        >>> logger.info("Pipeline started")
        >>>
        >>> # Logging with extra context
        >>> logger.info("Model trained", extra={
        ...     "model_name": "fraud_detector",
        ...     "accuracy": 0.95
        ... })
        >>>
        >>> # Using context manager
        >>> with LogContext(pipeline_id="pipeline_123"):
        ...     logger.info("Processing data")
        ...     # All logs in this block include pipeline_id
    """

    def __init__(self, name: str, correlation_id: Optional[str] = None) -> None:
        """Initialize the structured logger.

        Args:
            name: Logger name (typically __name__)
            correlation_id: Optional correlation ID (auto-generated if not provided)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.correlation_id = correlation_id or str(uuid.uuid4())

        # Set default formatter if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = ContextFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _get_context(self) -> Dict[str, Any]:
        """Get current log context.

        Returns:
            Dictionary with current context
        """
        context = _log_context.get().copy()
        context["correlation_id"] = self.correlation_id
        return context

    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Initialize the logging method.

        Args:
            level: Log level (logging.DEBUG, INFO, etc.)
            message: Log message
            extra: Optional extra context
            exc_info: Whether to include exception info
        """
        # Merge context with extra
        context = self._get_context()
        if extra:
            context.update(extra)

        # Log with context
        self.logger.log(level, message, extra={"context": context}, exc_info=exc_info)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message.

        Args:
            message: Log message
            extra: Optional extra context
        """
        self._log(logging.DEBUG, message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message.

        Args:
            message: Log message
            extra: Optional extra context
        """
        self._log(logging.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message.

        Args:
            message: Log message
            extra: Optional extra context
        """
        self._log(logging.WARNING, message, extra)

    def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
    ) -> None:
        """Log an error message.

        Args:
            message: Log message
            extra: Optional extra context
            exc_info: Whether to include exception info (default: True)
        """
        self._log(logging.ERROR, message, extra, exc_info)

    def critical(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
    ) -> None:
        """Log a critical message.

        Args:
            message: Log message
            extra: Optional extra context
            exc_info: Whether to include exception info (default: True)
        """
        self._log(logging.CRITICAL, message, extra, exc_info)

    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an exception with traceback.

        This should be called from an exception handler.

        Args:
            message: Log message
            extra: Optional extra context
        """
        self._log(logging.ERROR, message, extra, exc_info=True)

    def set_level(self, level: int) -> None:
        """Set the logging level.

        Args:
            level: Logging level (logging.DEBUG, INFO, etc.)
        """
        self.logger.setLevel(level)


class LogContext:
    """Context manager for adding temporary log context.

    This context manager allows you to add context fields that will be
    included in all log messages within the context. The context is
    automatically removed when exiting the context manager.

    Context is thread-safe and uses context variables, so each thread
    maintains its own context.

    Attributes:
        context: Dictionary of context fields to add

    Example:
        >>> logger = get_logger(__name__)
        >>>
        >>> with LogContext(pipeline_id="pipeline_123", user="john"):
        ...     logger.info("Starting pipeline")
        ...     # Log includes: pipeline_id=pipeline_123, user=john
        ...
        ...     with LogContext(step="ingestion"):
        ...         logger.info("Ingesting data")
        ...         # Log includes: pipeline_id, user, step=ingestion
        >>>
        >>> # Context is removed after exiting
        >>> logger.info("Pipeline complete")
    """

    def __init__(self, **context: Any) -> None:
        """Initialize the log context.

        Args:
            **context: Context fields as keyword arguments
        """
        self.context = context
        self._previous_context: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "LogContext":
        """Enter the context manager.

        Returns:
            Self for use in 'with' statement
        """
        # Save previous context
        self._previous_context = _log_context.get().copy()

        # Add new context
        current_context = _log_context.get().copy()
        current_context.update(self.context)
        _log_context.set(current_context)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Exit the context manager.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred

        Returns:
            False to propagate any exception
        """
        # Restore previous context
        if self._previous_context is not None:
            _log_context.set(self._previous_context)
        else:
            _log_context.set({})

        # Don't suppress exceptions
        return False


def get_logger(name: str, correlation_id: Optional[str] = None) -> StructuredLogger:
    """Get a structured logger instance.

    This is the main entry point for getting a logger. It returns a
    StructuredLogger instance with the given name.

    Args:
        name: Logger name (typically __name__)
        correlation_id: Optional correlation ID

    Returns:
        StructuredLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return StructuredLogger(name, correlation_id)


def set_global_context(**context: Any) -> None:
    """Set global log context.

    This function sets context that will be included in all log messages
    across all loggers in the current thread/async task.

    Args:
        **context: Context fields as keyword arguments

    Example:
        >>> set_global_context(environment="prod", version="1.0.0")
        >>> # All subsequent logs include environment and version
    """
    current_context = _log_context.get().copy()
    current_context.update(context)
    _log_context.set(current_context)


def clear_global_context() -> None:
    """Clear global log context.

    This function removes all global context fields.

    Example:
        >>> clear_global_context()
    """
    _log_context.set({})


def get_global_context() -> Dict[str, Any]:
    """Get current global log context.

    Returns:
        Dictionary with current global context

    Example:
        >>> context = get_global_context()
        >>> print(context)
    """
    return _log_context.get().copy()
