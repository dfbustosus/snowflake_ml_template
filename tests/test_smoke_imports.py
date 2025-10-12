"""Smoke tests for package imports and basic template renderers."""

from snowflake_ml_template.feature_store import api as fs_api
from snowflake_ml_template.infra import ddl_templates
from snowflake_ml_template.models.registry import ModelRegistryHelper
from snowflake_ml_template.pipelines import tasks as pipeline_tasks


def test_infra_templates_render():
    """Test that infra templates render expected SQL."""
    dbs = ddl_templates.render_environment_databases()
    assert "ML_DEV_DB" in dbs
    schemas = ddl_templates.render_schema_structure("ML_DEV_DB")
    assert "RAW_DATA" in schemas
    rbac = ddl_templates.render_rbac_roles()
    assert "CREATE ROLE IF NOT EXISTS ML_DATA_SCIENTIST" in rbac
    wh = ddl_templates.render_warehouses()
    assert "CREATE WAREHOUSE" in wh


def test_registry_promote_sql():
    """Test that model registry helper generates promote SQL."""
    helper = ModelRegistryHelper()
    sql = helper.promote_model(None, "ML_TEST_DB", "ML_PROD_DB", "CHURN", "v1")
    assert "CREATE MODEL" in sql and "ML_PROD_DB" in sql


def test_pipeline_task_templates():
    """Test that pipeline task templates render expected SQL."""
    t1 = pipeline_tasks.render_check_new_data_task()
    assert "CREATE OR REPLACE TASK" in t1
    t2 = pipeline_tasks.render_train_task()
    assert "CALL TRAIN_MODEL_PROC" in t2


def test_feature_store_helper_sanity():
    """Test that feature store helper can be instantiated."""
    # instantiate helper; methods are lazy and won't execute without a session
    helper = fs_api.FeatureStoreHelper()
    assert hasattr(helper, "register_entity")
