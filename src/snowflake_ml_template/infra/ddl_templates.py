"""DDL template generators for Snowflake environment setup.

These functions return SQL strings that can be executed by schemachange or
via a Snowflake connection. They intentionally don't execute anything.
"""

from typing import Iterable, List, Optional


def render_environment_databases(
    dev_db: str = "ML_DEV_DB", test_db: str = "ML_TEST_DB", prod_db: str = "ML_PROD_DB"
) -> str:
    """Return SQL to create the three-environment databases with cloning guidance."""
    sql = f"""
-- Create environment databases (idempotent style)
CREATE DATABASE IF NOT EXISTS {dev_db};
CREATE DATABASE IF NOT EXISTS {test_db};
CREATE DATABASE IF NOT EXISTS {prod_db};

-- Recommended: create ML_TEST_DB as a zero-copy clone of ML_PROD_DB when appropriate:
-- CREATE DATABASE {test_db} CLONE {prod_db};
"""
    return sql.strip()


def render_schema_structure(
    database: str = "ML_DEV_DB", schemas: Optional[Iterable[str]] = None
) -> str:
    """Return SQL creating the canonical schemas inside a database.

    Default schemas: RAW_DATA, FEATURES, MODELS, PIPELINES, ANALYTICS
    """
    if schemas is None:
        schemas = ["RAW_DATA", "FEATURES", "MODELS", "PIPELINES", "ANALYTICS"]

    parts: List[str] = [f"-- Schemas for {database}"]
    for s in schemas:
        parts.append(f"CREATE SCHEMA IF NOT EXISTS {database}.{s};")

    return "\n".join(parts)


def render_rbac_roles() -> str:
    """Return SQL that defines the persona-based roles and example grants.

    Note: This file provides a baseline. Principals (users) are intentionally not
    added here - that is an administrative action.
    """
    sql = """
-- Roles for MLOps personas
CREATE ROLE IF NOT EXISTS ML_DATA_SCIENTIST;
CREATE ROLE IF NOT EXISTS ML_ENGINEER;
CREATE ROLE IF NOT EXISTS ML_PROD_ADMIN;
CREATE ROLE IF NOT EXISTS ML_INFERENCE_SERVICE_ROLE;

-- Example grants (should be adapted by security team)
-- Grant dev DB full privileges to data scientists and engineers
GRANT ALL PRIVILEGES ON DATABASE ML_DEV_DB TO ROLE ML_DATA_SCIENTIST;
GRANT ALL PRIVILEGES ON DATABASE ML_DEV_DB TO ROLE ML_ENGINEER;

-- Grant limited access to production for data scientists (read-only)
GRANT USAGE ON DATABASE ML_PROD_DB TO ROLE ML_DATA_SCIENTIST;
GRANT USAGE ON SCHEMA ML_PROD_DB.RAW_DATA TO ROLE ML_DATA_SCIENTIST;
GRANT SELECT ON ALL TABLES IN SCHEMA ML_PROD_DB.RAW_DATA TO ROLE ML_DATA_SCIENTIST;

-- Grant engineers privileges in test DB for operational objects
GRANT ALL PRIVILEGES ON DATABASE ML_TEST_DB TO ROLE ML_ENGINEER;

-- Production admin is owner of prod DB objects (grants below are examples)
GRANT OWNERSHIP ON DATABASE ML_PROD_DB TO ROLE ML_PROD_ADMIN;

-- Inference service role: usage on specific model later during deployment
"""
    return sql.strip()


def render_warehouses() -> str:
    """Return SQL for recommended warehouse creation with suggested settings."""
    sql = """
-- Warehouses for ML workloads
CREATE WAREHOUSE IF NOT EXISTS INGEST_WH WITH
  WAREHOUSE_SIZE = 'X-SMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  MIN_CLUSTER_COUNT = 1
  MAX_CLUSTER_COUNT = 1;

CREATE WAREHOUSE IF NOT EXISTS TRANSFORM_WH WITH
  WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  MIN_CLUSTER_COUNT = 1
  MAX_CLUSTER_COUNT = 3;

-- Snowpark-optimized warehouse for training (single-node but large memory)
CREATE WAREHOUSE IF NOT EXISTS ML_TRAINING_WH WITH
  WAREHOUSE_SIZE = 'LARGE'
  WAREHOUSE_TYPE = 'SNOWPARK-OPTIMIZED'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  MIN_CLUSTER_COUNT = 1
  MAX_CLUSTER_COUNT = 1;

CREATE WAREHOUSE IF NOT EXISTS INFERENCE_WH WITH
  WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  MIN_CLUSTER_COUNT = 1
  MAX_CLUSTER_COUNT = 2;
"""
    return sql.strip()
