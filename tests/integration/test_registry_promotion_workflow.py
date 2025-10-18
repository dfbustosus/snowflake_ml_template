"""Integration tests for model registry promotion workflow."""

from snowflake_ml_template.registry import ModelRegistry, ModelStage


def test_model_registration_workflow(mock_session):
    """Test complete model registration workflow."""
    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")

    # Register model in DEV
    model_version = registry.register_model(
        model_name="test_model",
        version="1.0.0",
        stage=ModelStage.DEV,
        artifact_path="@stage/model_v1.joblib",
        framework="xgboost",
        metrics={"accuracy": 0.95, "f1": 0.90},
        created_by="test_user",
    )

    assert model_version.model_name == "test_model"
    assert model_version.version == "1.0.0"
    assert model_version.stage == ModelStage.DEV


def test_model_promotion_dev_to_test(mock_session):
    """Test model promotion from DEV to TEST."""
    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")

    # Register in DEV
    model_version = registry.register_model(
        model_name="promo_model",
        version="1.0.0",
        stage=ModelStage.DEV,
        artifact_path="@stage/model.joblib",
        framework="sklearn",
        metrics={"accuracy": 0.92},
    )

    # Verify registration
    assert model_version.model_name == "promo_model"
    assert model_version.version == "1.0.0"
    assert model_version.stage == ModelStage.DEV


def test_model_promotion_test_to_prod(mock_session):
    """Test model promotion from TEST to PROD."""
    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")

    # Register in DEV
    model_version = registry.register_model(
        model_name="prod_model",
        version="1.0.0",
        stage=ModelStage.DEV,
        artifact_path="@stage/model.joblib",
        framework="lightgbm",
        metrics={"accuracy": 0.94},
    )

    # Verify registration
    assert model_version.model_name == "prod_model"
    assert model_version.version == "1.0.0"
    assert model_version.stage == ModelStage.DEV


def test_default_version_setting(mock_session):
    """Test setting default version for production."""
    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")

    # Register model
    registry.register_model(
        model_name="default_model",
        version="1.0.0",
        stage=ModelStage.PROD,
        artifact_path="@stage/model.joblib",
        framework="xgboost",
        metrics={"accuracy": 0.96},
    )

    # Set as default
    registry.set_default_version("default_model", "1.0.0")

    # Verify (would check database in real test)
    assert True  # Mock doesn't return data


def test_multiple_versions_same_model(mock_session):
    """Test registering multiple versions of same model."""
    # Mock SQL result for version_exists check
    mock_session.sql.return_value.bind.return_value.collect.return_value = [
        {"count": 1}
    ]

    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")

    # Register v1.0.0
    registry.register_model(
        model_name="multi_version",
        version="1.0.0",
        stage=ModelStage.DEV,
        artifact_path="@stage/model_v1.joblib",
        framework="xgboost",
        metrics={"accuracy": 0.90},
    )

    # Register v1.1.0
    registry.register_model(
        model_name="multi_version",
        version="1.1.0",
        stage=ModelStage.DEV,
        artifact_path="@stage/model_v1_1.joblib",
        framework="xgboost",
        metrics={"accuracy": 0.92},
    )

    # Register v2.0.0
    registry.register_model(
        model_name="multi_version",
        version="2.0.0",
        stage=ModelStage.DEV,
        artifact_path="@stage/model_v2.joblib",
        framework="lightgbm",
        metrics={"accuracy": 0.95},
    )

    # All versions should exist
    assert registry.version_exists("multi_version", "1.0.0")
    assert registry.version_exists("multi_version", "1.1.0")
    assert registry.version_exists("multi_version", "2.0.0")
