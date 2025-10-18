"""Tests for deployment strategies."""

from snowflake_ml_template.core.base.deployment import (
    DeploymentConfig,
    DeploymentStrategy,
    DeploymentTarget,
)
from snowflake_ml_template.deployment.strategies.spcs import SPCSStrategy
from snowflake_ml_template.deployment.strategies.warehouse_udf import (
    WarehouseUDFStrategy,
)


def test_warehouse_udf_strategy_initialization():
    """Test UDF strategy initialization."""
    config = DeploymentConfig(
        strategy=DeploymentStrategy.WAREHOUSE_UDF,
        target=DeploymentTarget.BATCH,
        model_name="test",
        model_version="1.0.0",
        model_artifact_path="@stage/model.joblib",
        deployment_database="DB",
        deployment_schema="SCHEMA",
        deployment_name="test_udf",
        warehouse="WH",
    )
    strategy = WarehouseUDFStrategy(config)
    assert strategy.config.deployment_name == "test_udf"


def test_warehouse_udf_strategy_validation():
    """Test UDF strategy validation."""
    config = DeploymentConfig(
        strategy=DeploymentStrategy.WAREHOUSE_UDF,
        target=DeploymentTarget.BATCH,
        model_name="test",
        model_version="1.0.0",
        model_artifact_path="@stage/model.joblib",
        deployment_database="DB",
        deployment_schema="SCHEMA",
        deployment_name="test_udf",
        warehouse="WH",
    )
    strategy = WarehouseUDFStrategy(config)
    assert strategy.validate() is True


def test_spcs_strategy_initialization():
    """Test SPCS strategy initialization."""
    config = DeploymentConfig(
        strategy=DeploymentStrategy.SPCS,
        target=DeploymentTarget.REALTIME,
        model_name="test",
        model_version="1.0.0",
        model_artifact_path="@stage/model.joblib",
        deployment_database="DB",
        deployment_schema="SCHEMA",
        deployment_name="test_service",
        warehouse="WH",
        compute_pool="POOL",
    )
    strategy = SPCSStrategy(config)
    assert strategy.config.deployment_name == "test_service"


def test_spcs_strategy_validation():
    """Test SPCS strategy validation."""
    config = DeploymentConfig(
        strategy=DeploymentStrategy.SPCS,
        target=DeploymentTarget.REALTIME,
        model_name="test",
        model_version="1.0.0",
        model_artifact_path="@stage/model.joblib",
        deployment_database="DB",
        deployment_schema="SCHEMA",
        deployment_name="test_service",
        warehouse="WH",
        compute_pool="POOL",
    )
    strategy = SPCSStrategy(config)
    assert strategy.validate() is True
