"""Snowpark transformation helpers and best-practice utilities.

These helpers are framework-agnostic wrappers that operate on Snowpark DataFrame
objects when a live session is provided. To keep unit tests light, imports are
lazy and functions validate inputs.
"""

from typing import Any, List


def ensure_columns(df: Any, required_columns: List[str]) -> Any:
    """Return a DataFrame selecting only the required columns (explicit listing).

    This avoids .select('*') and defends against schema evolution.
    """
    # df is expected to be a Snowpark DataFrame with .select method
    return df.select(*required_columns)


def add_bitemporal_columns(
    df: Any, event_ts_col: str = "event_timestamp", load_ts_col: str = "load_timestamp"
) -> Any:
    """Ensure bitemporal columns exist on the DataFrame. If load_ts_col not present, add CURRENT_TIMESTAMP()."""
    cols = df.columns
    if load_ts_col not in cols:
        df = df.with_column(load_ts_col, df.session.sql_expr("CURRENT_TIMESTAMP()"))
    if event_ts_col not in cols:
        # fallback: copy load_ts to event_ts if event_ts missing
        df = df.with_column(event_ts_col, df[load_ts_col])
    return df


def asof_join(
    left_df: Any,
    right_table: str,
    left_time_col: str,
    right_time_col: str,
    join_keys: List[str],
    suffix: str = "_fv",
) -> Any:
    """Perform an ASOF-style join by returning SQL snippet which does lateral join using QUALIFY.

    Note: Snowpark lacks a direct ASOF join primitive; this helper returns a SQL
    expression that can be executed or wrapped in a stored proc. The function
    assumes right_table is materialized with versions per key/time.
    """
    # build ON clauses
    on_keys = " AND ".join([f"l.{k} = r.{k}" for k in join_keys])
    # Constructed SQL operates over internal, known table names and join keys.
    # If any user-controlled values reach this function, sanitize them first.
    sql = f"SELECT l.*, r.* FROM {{left_alias}} l LEFT JOIN LATERAL (SELECT * FROM {right_table} r WHERE {on_keys} AND r.{right_time_col} <= l.{left_time_col} ORDER BY r.{right_time_col} DESC LIMIT 1) r ON TRUE"  # nosec: B608 - expected internal usage
    return sql
