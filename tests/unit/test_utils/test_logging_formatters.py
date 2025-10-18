"""Unit tests for logging formatters."""

import json
import logging

from snowflake_ml_template.utils.logging.formatters import (
    ColorizedFormatter,
    ContextFormatter,
    JSONFormatter,
)


def test_json_formatter_basic_and_context_and_exception():
    """Test JSON formatter with basic and context and exception."""
    rec = logging.makeLogRecord(
        {
            "levelname": "INFO",
            "name": "t_json",
            "msg": "hello",
        }
    )
    fmt = JSONFormatter()
    out = fmt.format(rec)
    obj = json.loads(out)
    assert (
        obj["level"] == "INFO"
        and obj["logger"] == "t_json"
        and obj["message"] == "hello"
    )

    # with context and extra
    rec2 = logging.makeLogRecord(
        {
            "levelname": "ERROR",
            "name": "t_json",
            "msg": "oops",
            "context": {"a": 1},
            "extra_field": "x",
            "exc_info": (ValueError, ValueError("bad"), None),
        }
    )
    out2 = fmt.format(rec2)
    obj2 = json.loads(out2)
    assert obj2["context"] == {"a": 1} and obj2["extra_field"] == "x"


def test_context_formatter_with_and_without_context():
    """Test context formatter with and without context."""
    fmt = ContextFormatter()
    rec = logging.makeLogRecord({"levelname": "INFO", "name": "t_ctx", "msg": "hi"})
    out = fmt.format(rec)
    assert "hi" in out

    rec2 = logging.makeLogRecord(
        {"levelname": "INFO", "name": "t_ctx", "msg": "hi", "context": {"x": "y"}}
    )
    out2 = fmt.format(rec2)
    assert "x=y" in out2


def test_colorized_formatter_wraps_context_formatter():
    """Test colorized formatter wraps context formatter."""
    fmt = ColorizedFormatter()
    rec = logging.makeLogRecord({"levelname": "INFO", "name": "t_col", "msg": "msg"})
    out = fmt.format(rec)
    assert isinstance(out, str) and "msg" in out
