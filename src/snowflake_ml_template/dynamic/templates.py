"""Templates to create streams, dynamic tables and bi-temporal helpers."""

from typing import Optional


def render_stream_on_table(stream_name: str, source_table: str) -> str:
    """Render a CREATE STREAM ON TABLE statement for the given source table."""
    return f"CREATE OR REPLACE STREAM {stream_name} ON TABLE {source_table} SHOW_INITIAL_ROWS = TRUE;"


def render_dynamic_table(
    dynamic_name: str, select_sql: str, schedule: Optional[str] = None
) -> str:
    """Render SQL to create a Dynamic Table (or materialized-like table) from select_sql.

    If schedule is provided, include a refresh schedule comment (Dynamic Tables are declarative in Snowflake UI).
    """
    sched = f"-- Refresh schedule: {schedule}\n" if schedule else ""
    sql = f"{sched}CREATE OR REPLACE DYNAMIC TABLE {dynamic_name} AS {select_sql};"
    return sql
