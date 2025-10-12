"""Helpers to create a Snowpark session from environment variables.

This module avoids importing Snowflake packages until create_session_from_env is
called. It raises a helpful RuntimeError if the dependencies are missing.
"""

import os
from typing import Any, Dict, Optional


def create_session_from_env(extra: Optional[Dict[str, str]] = None) -> Any:
    """Create and return a Snowpark Session using environment variables.

    Required env vars: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
    """
    try:
        from snowflake.snowpark import Session
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "snowflake.snowpark is required to create a session"
        ) from exc

    cfg = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
    }

    if extra:
        cfg.update(extra)

    # remove None values
    cfg = {k: v for k, v in cfg.items() if v is not None}

    return Session.builder.configs(cfg).create()


def set_query_tag(session: Any, tag: str) -> None:
    """Set a query tag on a given session for cost attribution."""
    session.sql(f"ALTER SESSION SET QUERY_TAG = '{tag}'").collect()
