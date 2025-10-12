"""SQL template generators for ingestion patterns: Snowpipe, COPY INTO, Streaming."""

from typing import Optional


def render_create_pipe(
    pipe_name: str,
    stage: str,
    file_format: str = "(TYPE = 'PARQUET')",
    copy_options: Optional[str] = None,
) -> str:
    """Return SQL to create a Snowpipe that automates COPY INTO from a stage.

    This template is intentionally generic — the operator should set notifications
    in the cloud provider and grant rights to the notification integration.
    """
    copy_opts = (
        f"COPY INTO {pipe_name}_target FROM {stage} FILE_FORMAT = {file_format} "
    )
    if copy_options:
        copy_opts += copy_options

    sql = f"""
-- Snowpipe: create target table, file format and pipe (idempotent)
CREATE OR REPLACE FILE FORMAT {pipe_name}_fmt {file_format};

-- The target table must exist prior to creating the pipe. Example COPY command:
-- {copy_opts};

CREATE OR REPLACE PIPE {pipe_name} AS
  COPY INTO {pipe_name}_target
  FROM {stage}
  FILE_FORMAT = {pipe_name}_fmt
  ON_ERROR = 'CONTINUE';
"""
    return sql.strip()


def render_copy_into(
    table: str,
    stage: str,
    file_format: str = "(TYPE='PARQUET')",
    pattern: Optional[str] = None,
    force: bool = False,
) -> str:
    r"""Return a COPY INTO SQL statement for bulk loads.

    Use pattern to limit files (e.g., r'.*\.parquet'). Set FORCE=TRUE for re-ingests.
    """
    pattern_clause = f"PATTERN = '{pattern}'" if pattern else ""
    force_clause = "FORCE=TRUE" if force else ""
    sql = f"COPY INTO {table} FROM {stage} FILE_FORMAT = {file_format} {pattern_clause} {force_clause};"
    return sql


def render_create_streaming_table(
    table_name: str, columns: str = "(data VARIANT, load_ts TIMESTAMP_LTZ)"
) -> str:
    """Render SQL to create a table optimized for Snowpipe Streaming ingestion.

    The table uses VARIANT to accept flexible JSON-ish rows and a load timestamp.
    """
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} {columns};"
    return sql


def render_infer_schema_and_create(
    table_name: str, stage: str, path_pattern: str = "*.parquet"
) -> str:
    """Return SQL that infers schema from staged files and creates a table using TEMPLATE.

    This uses INFER_SCHEMA over a set of staged files and then shows how to create a table
    using the inferred template. This is illustrative and operators should verify types.
    """
    sql = f"""
-- Infer schema from sample files on stage
SELECT SYSTEM$INFER_SCHEMA('{stage}', PATTERN => '{path_pattern}');

-- The output above can be inspected and used to create a table. Example using CREATE TABLE ... USING TEMPLATE
-- CREATE TABLE {table_name} USING TEMPLATE (<inferred-template-json>);
"""
    return sql.strip()


def render_pipe_status_query(pipe_name: str) -> str:
    """Return SQL to query the status of a Snowpipe and recent errors."""
    sql = f"SELECT SYSTEM$PIPE_STATUS('{pipe_name}');"
    return sql


def render_copy_history_query(table_name: str, since_hours: int = 24) -> str:
    """Query COPY_HISTORY for troubleshooting recent loads.

    Note: this builds SQL for administrative troubleshooting; inputs should be
    trusted (internal table names). If table_name can be external, sanitize
    or parameterize accordingly.
    """
    sql = f"SELECT * FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(TABLE_NAME => '{table_name}', START_TIME => DATEADD(hour, -{since_hours}, CURRENT_TIMESTAMP())));"  # nosec: B608 - table_name is expected to be an internal identifier
    return sql


def render_streaming_row_sproc(
    proc_name: str = "INSERT_STREAM_ROW_PROC", table_name: str = "STREAMING_RAW"
) -> str:
    """Generate and return a streaming row stored procedure template.

    This template accepts a JSON row and inserts into a streaming table.
    This models Snowpipe Streaming ingestion patterns where client apps write rows via API and a SProc persists them.
    """
    sql = f"""
CREATE OR REPLACE PROCEDURE {proc_name}(payload VARIANT)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
HANDLER = 'run'
AS
$$
def run(session, payload):
    # payload is expected to be a VARIANT-compatible JSON structure
    session.sql("INSERT INTO {table_name}(data, load_ts) SELECT PARSE_JSON(%s), CURRENT_TIMESTAMP()", [str(payload)]).collect()
    return 'ok'
$$;
"""
    return sql.strip()
