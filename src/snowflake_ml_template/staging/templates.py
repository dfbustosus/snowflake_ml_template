"""Templates for staging raw semi-structured data into VARIANT and parser SProcs."""


def render_variant_stage_table(table_name: str = "RAW_DATA.STAGE") -> str:
    """Return SQL to create a VARIANT-based staging table for raw data."""
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (data VARIANT, load_timestamp TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP());"
    return sql


def render_parser_sproc(
    proc_name: str = "PARSE_AND_FLATTEN_PROC",
    source_table: str = "RAW_DATA.STAGE",
    target_table: str = "RAW_DATA.CLEANED",
) -> str:
    """Return a Python stored procedure template that parses VARIANT rows and writes to a structured table."""
    sql = f"""
CREATE OR REPLACE PROCEDURE {proc_name}()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python',)
HANDLER = 'run'
AS
$$
def run(session):
    # Example: read staging table and flatten into structured table
    df = session.table('{source_table}')
    # TODO: replace with transformation logic that extracts fields from VARIANT
    # Example: df = df.select(df['data']:id.as_('id'), ...)
    df.write.save_as_table('{target_table}', mode='append')
    return 'ok'
$$;
"""
    return sql.strip()
