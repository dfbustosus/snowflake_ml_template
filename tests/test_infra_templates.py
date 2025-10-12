"""Tests for infra DDL template generators."""

from snowflake_ml_template.infra import ddl_templates


def test_render_env_databases_contains_three():
    """Test that render_environment_databases includes all three environment databases."""
    sql = ddl_templates.render_environment_databases()
    assert "ML_DEV_DB" in sql
    assert "ML_TEST_DB" in sql
    assert "ML_PROD_DB" in sql


def test_render_schema_structure_default():
    """Test that render_schema_structure creates default schemas for the given database."""
    sql = ddl_templates.render_schema_structure("ML_DEV_DB")
    assert "RAW_DATA" in sql
    assert "FEATURES" in sql
    assert "MODELS" in sql
