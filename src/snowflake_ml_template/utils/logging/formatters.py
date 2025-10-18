"""Log formatters for structured logging.

This module provides custom log formatters that support:
- JSON output for machine parsing
- Human-readable text output with context
- Colorized output for console

Classes:
    JSONFormatter: Format logs as JSON
    ContextFormatter: Format logs with context fields
    ColorizedFormatter: Format logs with colors (for console)
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Format log records as JSON.

    This formatter outputs log records as JSON objects, making them
    easy to parse and ingest into log aggregation systems.

    Each log record includes:
    - timestamp: ISO 8601 timestamp
    - level: Log level name
    - logger: Logger name
    - message: Log message
    - context: Context fields
    - exception: Exception info (if present)

    Example output:
        {
            "timestamp": "2025-10-16T21:30:45.123456",
            "level": "INFO",
            "logger": "pipeline.fraud_detection",
            "message": "Pipeline started",
            "context": {
                "correlation_id": "abc-123",
                "pipeline_id": "pipeline_456"
            }
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON string
        """
        # Build log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context if present
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "context",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


class ContextFormatter(logging.Formatter):
    """Format log records with context fields.

    This formatter outputs human-readable log messages with context
    fields appended. It's suitable for console output and log files.

    Example output:
        2025-10-16 21:30:45,123 - INFO - pipeline.fraud_detection - Pipeline started [correlation_id=abc-123, pipeline_id=pipeline_456]
    """

    def __init__(
        self,
        fmt: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """Initialize the formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
        """
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with context.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Format base message
        base_message = super().format(record)

        # Add context if present
        if hasattr(record, "context") and record.context:
            context_str = ", ".join(f"{k}={v}" for k, v in record.context.items())
            return f"{base_message} [{context_str}]"

        return base_message


class ColorizedFormatter(ContextFormatter):
    """Format log records with colors for console output.

    This formatter adds ANSI color codes to log messages based on
    the log level, making them easier to read in the console.

    Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red + Bold
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Red + Bold
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted log string with ANSI color codes
        """
        # Get color for this level
        color = self.COLORS.get(record.levelname, "")

        # Format message
        message = super().format(record)

        # Add color
        if color:
            return f"{color}{message}{self.RESET}"

        return message
