"""Tests for Phase 2 SQL template generators (ingest, staging, transform)."""

from snowflake_ml_template.dynamic.templates import (
    render_dynamic_table,
    render_stream_on_table,
)
from snowflake_ml_template.ingest.templates import (
    render_copy_into,
    render_create_pipe,
    render_create_streaming_table,
)
from snowflake_ml_template.staging.templates import (
    render_parser_sproc,
    render_variant_stage_table,
)
from snowflake_ml_template.transform.helpers import (
    add_bitemporal_columns,
    asof_join,
    ensure_columns,
)


def test_ingest_templates_basic():
    """Test basic functionality of ingest template generators."""
    p = render_create_pipe("pipe1", "@mystage")
    assert "CREATE OR REPLACE PIPE pipe1" in p or "CREATE OR REPLACE PIPE" in p
    c = render_copy_into("mytable", "@mystage", pattern=r".*\\.parquet")
    assert "COPY INTO mytable" in c
    s = render_create_streaming_table("stream_tbl")
    assert "CREATE TABLE IF NOT EXISTS stream_tbl" in s


def test_staging_templates_basic():
    """Test basic functionality of staging template generators."""
    v = render_variant_stage_table("RAW_DATA.STAGE")
    assert "CREATE TABLE IF NOT EXISTS RAW_DATA.STAGE" in v
    sp = render_parser_sproc()
    assert "CREATE OR REPLACE PROCEDURE" in sp


def test_dynamic_templates_basic():
    """Test basic functionality of dynamic table template generators."""
    st = render_stream_on_table("mystream", "RAW_DATA.STAGE")
    assert "CREATE OR REPLACE STREAM mystream" in st
    dt = render_dynamic_table("DT_CUSTOMER", "SELECT 1")
    assert "CREATE OR REPLACE DYNAMIC TABLE DT_CUSTOMER" in dt


def test_transform_helpers_sql_snippet():
    """Test that transform helpers generate expected SQL snippets."""
    sql = asof_join(
        left_df=None,
        right_table="FEATURES.customer_fv",
        left_time_col="evt_ts",
        right_time_col="fv_ts",
        join_keys=["customer_id"],
    )
    assert "SELECT l.*, r.* FROM" in sql
    # ensure ensure_columns and add_bitemporal_columns are importable (can't execute without Snowpark df)
    assert callable(ensure_columns)
    assert callable(add_bitemporal_columns)
