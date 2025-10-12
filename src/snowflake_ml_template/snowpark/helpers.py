"""Small Snowpark migration helpers with lazy imports.

These helpers are intentionally lightweight and import Snowpark only when needed so
unit tests can run in environments without Snowpark installed.
"""

from typing import Any


def import_snowpark() -> tuple[Any, Any]:
    """Lazy import Snowpark types or raise a helpful error if unavailable."""
    try:
        from snowflake.snowpark import DataFrame as _DataFrame
        from snowflake.snowpark import Session as _Session

        return _DataFrame, _Session
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Snowpark is required for this operation") from exc


def pandas_to_snowpark(session: Any, pdf: Any) -> Any:
    """Convert a pandas DataFrame to a Snowpark DataFrame.

    Args:
        session: Snowpark Session or compatible object with `create_dataframe`.
        pdf: pandas.DataFrame

    Returns:
        Snowpark DataFrame-like object.
    """
    # avoid importing snowpark at module import time; rely on duck-typing for tests
    if not hasattr(session, "create_dataframe"):
        raise TypeError("session must provide create_dataframe(pdf) method")
    return session.create_dataframe(pdf)


def snowpark_to_pandas(sdf: Any) -> Any:
    """Convert a Snowpark DataFrame to pandas DataFrame using to_pandas().

    The function will call `to_pandas()` on the Snowpark DF. It will raise a
    helpful error if the method is not available.
    """
    if not hasattr(sdf, "to_pandas"):
        raise TypeError("sdf must provide to_pandas() method")
    return sdf.to_pandas()
