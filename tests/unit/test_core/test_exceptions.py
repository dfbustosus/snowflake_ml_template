"""Unit tests for exceptions."""

from snowflake_ml_template.core.exceptions.errors import (
    AlertingError,
    ConfigurationError,
    DataPreparationError,
    DataValidationError,
    DeploymentError,
    DeploymentValidationError,
    DriftDetectionError,
    EntityNotFoundError,
    FeatureStoreError,
    FeatureVersionError,
    FeatureViewNotFoundError,
    IngestionError,
    MLOpsError,
    ModelNotFoundError,
    ModelTrainingError,
    MonitoringError,
    PipelineError,
    PipelineExecutionError,
    PipelineValidationError,
    RegistryError,
    ServiceUnavailableError,
    SessionError,
    SourceConnectionError,
    TrainingError,
    TransformationError,
    VersionNotFoundError,
)


def test_mlops_error_builds_full_message_with_context_and_cause():
    """Test mlops error builds full message with context and cause."""
    original = ValueError("bad")
    err = MLOpsError("base", context={"a": 1, "b": 2}, original_error=original)
    msg = str(err)
    assert "base" in msg
    assert "a=1" in msg and "b=2" in msg
    assert "bad" in msg


def test_simple_exception_hierarchy_instantiation():
    """Test simple exception hierarchy instantiation."""
    # Base groups
    assert isinstance(ConfigurationError("x"), MLOpsError)
    assert isinstance(SessionError("x"), MLOpsError)
    assert isinstance(PipelineError("x"), MLOpsError)
    assert isinstance(IngestionError("x"), MLOpsError)
    assert isinstance(TransformationError("x"), MLOpsError)
    assert isinstance(FeatureStoreError("x"), MLOpsError)
    assert isinstance(TrainingError("x"), MLOpsError)
    assert isinstance(RegistryError("x"), MLOpsError)
    assert isinstance(DeploymentError("x"), MLOpsError)
    assert isinstance(MonitoringError("x"), MLOpsError)

    # Specifics
    assert isinstance(PipelineValidationError("x"), PipelineError)
    assert isinstance(PipelineExecutionError("x"), PipelineError)
    assert isinstance(SourceConnectionError("x"), IngestionError)
    assert isinstance(DataValidationError("x"), IngestionError)
    assert isinstance(FeatureVersionError("x"), FeatureStoreError)
    assert isinstance(DataPreparationError("x"), TrainingError)
    assert isinstance(ModelTrainingError("x"), TrainingError)
    assert isinstance(DeploymentValidationError("x"), DeploymentError)
    assert isinstance(ServiceUnavailableError("x"), DeploymentError)
    assert isinstance(DriftDetectionError("x"), MonitoringError)
    assert isinstance(AlertingError("x"), MonitoringError)


def test_entity_not_found_error_has_entity_name():
    """Test entity not found error has entity name."""
    e = EntityNotFoundError("Customer", context={"id": 1})
    assert "Entity not found: Customer" in str(e)
    assert e.entity_name == "Customer"


def test_feature_view_not_found_error_has_name():
    """Test feature view not found error has name."""
    e = FeatureViewNotFoundError("fv_name")
    assert "Feature view not found: fv_name" in str(e)
    assert e.feature_view_name == "fv_name"


def test_model_not_found_error_has_name():
    """Test model not found error has name."""
    e = ModelNotFoundError("churn")
    assert "Model not found: churn" in str(e)
    assert e.model_name == "churn"


def test_version_not_found_error_has_model_and_version():
    """Test version not found error has model and version."""
    e = VersionNotFoundError("churn", "1")
    assert "Version not found: churn v1" in str(e)
    assert e.model_name == "churn" and e.version == "1"
