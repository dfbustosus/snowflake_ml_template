"""Tests for Pipeline Template."""

import pytest

from snowflake_ml_template.core.base.pipeline import PipelineConfig
from snowflake_ml_template.pipelines._base.template import PipelineTemplate


class SimplePipeline(PipelineTemplate):
    """Simple pipeline implementation for testing."""

    def engineer_features(self):
        """Test feature engineering."""
        pass

    def train_model(self):
        """Test model training."""
        pass


def test_pipeline_initialization(mock_session):
    """Test pipeline initialization."""
    config = PipelineConfig(
        name="test_pipeline",
        version="1.0.0",
        environment="dev",
        database="TEST_DB",
        warehouse="TEST_WH",
    )

    pipeline = SimplePipeline(mock_session, config)

    assert pipeline.session == mock_session
    assert pipeline.config.name == "test_pipeline"
    assert pipeline.feature_store is not None
    assert pipeline.model_registry is not None


def test_pipeline_validation_fails_without_name(mock_session):
    """Test pipeline validation fails without name."""
    with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
        PipelineConfig(
            name="",
            version="1.0.0",
            environment="dev",
            database="TEST_DB",
            warehouse="TEST_WH",
        )
