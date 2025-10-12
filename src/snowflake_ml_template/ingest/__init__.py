"""Ingestion helpers: generate SQL for Snowpipe, COPY INTO, and streaming ingestion.

These helpers return idempotent SQL statements and guidance strings. They do not
execute anything — execution should be done by schemachange or via a Snowpark
session in a deployment pipeline.
"""

from .templates import (
    render_copy_into,
    render_create_pipe,
    render_create_streaming_table,
)

__all__ = ["render_create_pipe", "render_copy_into", "render_create_streaming_table"]
