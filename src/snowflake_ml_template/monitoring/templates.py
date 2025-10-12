"""Monitoring SQL snippets for Snowpipe and load/ingest health checks."""


def render_pipe_status(pipe_name: str) -> str:
    """Return SQL to check Snowpipe status via SYSTEM$PIPE_STATUS."""
    return f"SELECT SYSTEM$PIPE_STATUS('{pipe_name}');"


def render_recent_load_errors(since_minutes: int = 60) -> str:
    """Return SQL to query recent load errors from LOAD_HISTORY/INFORMATION_SCHEMA views.

    This example uses ACCOUNT_USAGE or INFORMATION_SCHEMA depending on available privileges.
    """
    sql = f"SELECT * FROM TABLE(INFORMATION_SCHEMA.LOAD_HISTORY(START_TIME => DATEADD(minute, -{since_minutes}, CURRENT_TIMESTAMP())));"  # nosec: B608 - internal monitoring SQL; ensure inputs are trusted
    return sql
