"""Integration tests for end-to-end pipeline execution."""

from snowflake_ml_template.core.base.pipeline import PipelineConfig
from snowflake_ml_template.pipelines._base.template import PipelineTemplate


class IntegrationTestPipeline(PipelineTemplate):
    """Integration test pipeline."""

    def engineer_features(self):
        """Engineer test features."""
        # Skip feature engineering in integration test
        # (requires real Snowflake session)
        pass

    def train_model(self):
        """Train test model."""
        # Skip training in integration test
        # (requires real data)
        pass


def test_full_pipeline_execution(mock_session):
    """Test complete pipeline execution from start to finish."""
    config = PipelineConfig(
        name="integration_test_pipeline",
        version="1.0.0",
        environment="dev",
        database="TEST_DB",
        warehouse="TEST_WH",
    )

    pipeline = IntegrationTestPipeline(mock_session, config)

    # Execute full pipeline
    result = pipeline.execute()

    # Verify execution
    assert result.status == "success"
    assert result.pipeline_name == "integration_test_pipeline"
    assert len(result.stages_completed) > 0


def test_feature_store_integration(mock_session):
    """Test Feature Store integration."""
    config = PipelineConfig(
        name="feature_store_test",
        version="1.0.0",
        environment="dev",
        database="TEST_DB",
        warehouse="TEST_WH",
    )

    pipeline = IntegrationTestPipeline(mock_session, config)

    # Verify feature store initialized
    assert pipeline.feature_store is not None
    assert pipeline.feature_store.database == "TEST_DB"


def test_model_registry_integration(mock_session):
    """Test Model Registry integration."""
    config = PipelineConfig(
        name="registry_test",
        version="1.0.0",
        environment="dev",
        database="TEST_DB",
        warehouse="TEST_WH",
    )

    pipeline = IntegrationTestPipeline(mock_session, config)

    # Verify registry initialized
    assert pipeline.model_registry is not None
    assert pipeline.model_registry.database == "TEST_DB"


def test_training_orchestrator_integration(mock_session):
    """Test Training Orchestrator integration."""
    config = PipelineConfig(
        name="training_test",
        version="1.0.0",
        environment="dev",
        database="TEST_DB",
        warehouse="TEST_WH",
    )

    pipeline = IntegrationTestPipeline(mock_session, config)

    # Verify training orchestrator initialized
    assert pipeline.training_orch is not None


def test_deployment_orchestrator_integration(mock_session):
    """Test Deployment Orchestrator integration."""
    config = PipelineConfig(
        name="deployment_test",
        version="1.0.0",
        environment="dev",
        database="TEST_DB",
        warehouse="TEST_WH",
    )

    pipeline = IntegrationTestPipeline(mock_session, config)

    # Verify deployment orchestrator initialized
    assert pipeline.deployment_orch is not None
