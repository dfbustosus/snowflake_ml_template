"""Tests for logging utilities."""

from snowflake_ml_template.utils.logging import get_logger


def test_get_logger():
    """Test getting a logger."""
    logger = get_logger("test_module")
    assert logger is not None


def test_get_logger_with_correlation_id():
    """Test getting logger with correlation ID."""
    logger = get_logger("test_module", correlation_id="test-123")
    assert logger is not None


def test_structured_logger_info():
    """Test structured logger info."""
    logger = get_logger("test")
    logger.info("Test message")
    assert True


def test_structured_logger_error():
    """Test structured logger error."""
    logger = get_logger("test")
    logger.error("Error message")
    assert True


def test_structured_logger_warning():
    """Test structured logger warning."""
    logger = get_logger("test")
    logger.warning("Warning message")
    assert True


def test_structured_logger_debug():
    """Test structured logger debug."""
    logger = get_logger("test")
    logger.debug("Debug message")
    assert True
