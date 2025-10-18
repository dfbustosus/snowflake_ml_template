"""Tests for model registry."""

from snowflake_ml_template.registry import ModelRegistry, ModelStage, ModelVersion


def test_model_registry_initialization(mock_session):
    """Test model registry initialization."""
    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")
    assert registry.session == mock_session
    assert registry.database == "ML_DB"
    assert registry.schema == "MODELS"


def test_model_version_creation():
    """Test model version creation."""
    version = ModelVersion(
        model_name="test_model",
        version="1.0.0",
        stage=ModelStage.DEV,
        artifact_path="@stage/model.joblib",
        framework="xgboost",
        metrics={"accuracy": 0.95},
    )
    assert version.model_name == "test_model"
    assert version.version == "1.0.0"
    assert version.stage == ModelStage.DEV
    assert version.metrics["accuracy"] == 0.95


def test_model_stage_enum():
    """Test model stage enum."""
    assert ModelStage.DEV.value == "dev"
    assert ModelStage.TEST.value == "test"
    assert ModelStage.PROD.value == "prod"


def test_register_model_returns_version(mock_session):
    """Test register model returns ModelVersion."""
    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")

    version = registry.register_model(
        model_name="test_model",
        version="1.0.0",
        stage=ModelStage.DEV,
        artifact_path="@stage/model.joblib",
        framework="xgboost",
        metrics={"accuracy": 0.95},
    )

    assert isinstance(version, ModelVersion)
    assert version.model_name == "test_model"


def test_set_default_version(mock_session):
    """Test setting default version."""
    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")

    # Should not raise
    registry.set_default_version("test_model", "1.0.0")
    assert True


def test_version_exists_true(mock_session):
    """Test version exists returns true."""
    mock_session.sql.return_value.bind.return_value.collect.return_value = [
        {"count": 1}
    ]

    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")
    result = registry.version_exists("test_model", "1.0.0")

    assert result is True


def test_version_exists_false(mock_session):
    """Test version exists returns false."""
    mock_session.sql.return_value.bind.return_value.collect.return_value = [
        {"count": 0}
    ]

    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")
    result = registry.version_exists("test_model", "1.0.0")

    assert result is False


def test_version_exists_no_result(mock_session):
    """Test version exists with no result."""
    mock_session.sql.return_value.bind.return_value.collect.return_value = []

    registry = ModelRegistry(mock_session, "ML_DB", "MODELS")
    result = registry.version_exists("test_model", "1.0.0")

    assert result is False
