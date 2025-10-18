"""Structured logging system for the MLOps framework.

This module provides a structured logging system with:
- Correlation IDs for request tracking
- Context injection (pipeline_id, model_name, etc.)
- Multiple output formats (JSON, text)
- Integration with Snowflake event tables
- Log level management

Classes:
    StructuredLogger: Main logger with context support
    LogContext: Context manager for adding log context
"""

from snowflake_ml_template.utils.logging.formatters import (
    ContextFormatter,
    JSONFormatter,
)
from snowflake_ml_template.utils.logging.logger import (
    LogContext,
    StructuredLogger,
    get_logger,
)

__all__ = [
    "StructuredLogger",
    "LogContext",
    "get_logger",
    "JSONFormatter",
    "ContextFormatter",
]
