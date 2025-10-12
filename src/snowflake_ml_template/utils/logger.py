"""Logger setup with custom formatting and handlers."""

import logging
import os
import sys
from datetime import datetime


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and timestamps."""

    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and timestamp.

        Args:
            record: The log record to format.

        Returns:
            The formatted log string including a timestamp.
        """
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        # Format with standard attributes first
        formatted_message = super().format(record)

        # Add timestamp directly to the formatted message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} | {formatted_message}"


def setup_logger(name: str = "ua_leads") -> logging.Logger:
    """Set up a logger with both file and console handlers.

    Args:
        name (str): Name of the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only set up handlers if they haven't been set up already
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Try to create file handler, but fallback to console-only if it fails
        # This handles cases where code runs from a ZIP file (e.g., Spark executors)
        try:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

            # Check if we're running from a ZIP file (Spark executors)
            if not __file__.endswith(".zip"):
                os.makedirs(logs_dir, exist_ok=True)

                # File handler - logs everything to a file
                log_file = os.path.join(logs_dir, f"{name}.log")
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(
                    "%(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
        except (OSError, NotADirectoryError):
            # Running from ZIP or no write permissions - console only
            # This is expected in Spark executors, so don't fail
            pass

        # Console handler - with colors (always add this)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = CustomFormatter("%(levelname)s | %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add console handler to logger
        logger.addHandler(console_handler)

    return logger


# Create a default logger instance
logger = setup_logger()
