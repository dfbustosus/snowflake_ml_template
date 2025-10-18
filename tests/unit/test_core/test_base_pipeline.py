"""Unit tests for BasePipeline and related classes.

These tests validate the core pipeline abstraction without requiring
a Snowflake connection. They use mocks to simulate Snowflake sessions
and test the Template Method pattern implementation.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock

import pytest

from snowflake_ml_template.core.base.pipeline import (
    BasePipeline,
    PipelineConfig,
    PipelineExecutionStatus,
    PipelineResult,
    PipelineStage,
)


class RecordingTracker:
    """Simple tracker for capturing emitted events."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.events = []

    def record_event(self, component: str, event: str, payload: dict) -> None:
        """Record an event."""
        self.events.append((component, event, payload))


class TestPipelineConfig:
    """Test cases for PipelineConfig dataclass."""

    def test_valid_config(self) -> None:
        """Test creating a valid pipeline configuration."""
        config = PipelineConfig(
            name="test_pipeline",
            version="1.0.0",
            environment="dev",
            database="ML_DEV_DB",
            warehouse="ML_TRAINING_WH",
        )

        assert config.name == "test_pipeline"
        assert config.version == "1.0.0"
        assert config.environment == "dev"
        assert config.database == "ML_DEV_DB"
        assert config.warehouse == "ML_TRAINING_WH"

    def test_empty_name_raises_error(self) -> None:
        """Test that empty pipeline name raises ValueError."""
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            PipelineConfig(
                name="",
                version="1.0.0",
                environment="dev",
                database="ML_DEV_DB",
                warehouse="ML_TRAINING_WH",
            )

    def test_empty_version_raises_error(self) -> None:
        """Test that empty version raises ValueError."""
        with pytest.raises(ValueError, match="Pipeline version cannot be empty"):
            PipelineConfig(
                name="test_pipeline",
                version="",
                environment="dev",
                database="ML_DEV_DB",
                warehouse="ML_TRAINING_WH",
            )

    def test_invalid_environment_raises_error(self) -> None:
        """Test that invalid environment raises ValueError."""
        with pytest.raises(ValueError, match="Invalid environment"):
            PipelineConfig(
                name="test_pipeline",
                version="1.0.0",
                environment="invalid",
                database="ML_DEV_DB",
                warehouse="ML_TRAINING_WH",
            )

    def test_valid_environments(self) -> None:
        """Test that all valid environments are accepted."""
        for env in ["dev", "test", "prod"]:
            config = PipelineConfig(
                name="test_pipeline",
                version="1.0.0",
                environment=env,
                database="ML_DEV_DB",
                warehouse="ML_TRAINING_WH",
            )
            assert config.environment == env


class TestPipelineResult:
    """Test cases for PipelineResult dataclass."""

    def test_successful_result(self) -> None:
        """Test creating a successful pipeline result."""
        result = PipelineResult(
            status="success",
            pipeline_name="test_pipeline",
            pipeline_version="1.0.0",
            stages_completed=["ingestion", "transformation", "training"],
        )

        assert result.status == "success"
        assert result.pipeline_name == "test_pipeline"
        assert result.pipeline_version == "1.0.0"
        assert len(result.stages_completed) == 3
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test creating a failed pipeline result."""
        result = PipelineResult(
            status="failed",
            pipeline_name="test_pipeline",
            pipeline_version="1.0.0",
            stages_completed=["ingestion"],
            error="Training failed: insufficient data",
        )

        assert result.status == "failed"
        assert result.error is not None
        assert "Training failed" in result.error

    def test_duration_calculation(self) -> None:
        """Test that duration is calculated correctly."""
        start = datetime(2025, 1, 1, 10, 0, 0)
        end = datetime(2025, 1, 1, 10, 5, 30)

        result = PipelineResult(status="success", start_time=start, end_time=end)

        assert result.duration_seconds == 330.0  # 5 minutes 30 seconds


class TestBasePipeline:
    """Test cases for BasePipeline abstract class."""

    @pytest.fixture
    def mock_session(self) -> Mock:
        """Create a mock Snowflake session."""
        session = Mock()
        session.sql = MagicMock(return_value=Mock())
        return session

    @pytest.fixture
    def valid_config(self) -> PipelineConfig:
        """Create a valid pipeline configuration."""
        return PipelineConfig(
            name="test_pipeline",
            version="1.0.0",
            environment="dev",
            database="ML_DEV_DB",
            warehouse="ML_TRAINING_WH",
        )

    @pytest.fixture
    def concrete_pipeline(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> BasePipeline:
        """Create a concrete implementation of BasePipeline for testing."""

        class ConcretePipeline(BasePipeline):
            """Concrete pipeline implementation for testing."""

            def engineer_features(self) -> None:
                """Mock feature engineering."""
                self.logger.info("Engineering features")

            def train_model(self) -> None:
                """Mock model training."""
                self.logger.info("Training model")

        return ConcretePipeline(mock_session, valid_config)

    def test_initialization_with_valid_inputs(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> None:
        """Test that pipeline initializes correctly with valid inputs."""

        class TestPipeline(BasePipeline):
            def engineer_features(self) -> None:
                pass

            def train_model(self) -> None:
                pass

        pipeline = TestPipeline(mock_session, valid_config)

        assert pipeline.session == mock_session
        assert pipeline.config == valid_config
        assert pipeline._current_stage is None
        assert pipeline._start_time is None
        assert pipeline._stages_completed == []

    def test_initialization_with_none_session_raises_error(
        self, valid_config: PipelineConfig
    ) -> None:
        """Test that None session raises ValueError."""

        class TestPipeline(BasePipeline):
            def engineer_features(self) -> None:
                pass

            def train_model(self) -> None:
                pass

        with pytest.raises(ValueError, match="Session cannot be None"):
            TestPipeline(None, valid_config)

    def test_initialization_with_none_config_raises_error(
        self, mock_session: Mock
    ) -> None:
        """Test that None config raises ValueError."""

        class TestPipeline(BasePipeline):
            def engineer_features(self) -> None:
                pass

            def train_model(self) -> None:
                pass

        with pytest.raises(ValueError, match="Config cannot be None"):
            TestPipeline(mock_session, None)

    def test_successful_pipeline_execution(
        self, concrete_pipeline: BasePipeline
    ) -> None:
        """Test that pipeline executes all stages successfully."""
        result = concrete_pipeline.execute()

        assert result.status == "success"
        assert result.pipeline_name == "test_pipeline"
        assert result.pipeline_version == "1.0.0"
        assert result.error is None
        assert len(result.stages_completed) == 9  # All 9 stages
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.duration_seconds > 0

    def test_pipeline_execution_with_feature_engineering_failure(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> None:
        """Test that pipeline handles feature engineering failure correctly."""

        class FailingPipeline(BasePipeline):
            def engineer_features(self) -> None:
                raise ValueError("Feature engineering failed")

            def train_model(self) -> None:
                pass

        pipeline = FailingPipeline(mock_session, valid_config)
        result = pipeline.execute()

        assert result.status == "failed"
        assert result.error is not None
        assert "Feature engineering failed" in result.error
        assert PipelineStage.FEATURE_ENGINEERING.value in result.error

    def test_pipeline_execution_with_training_failure(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> None:
        """Test that pipeline handles training failure correctly."""

        class FailingPipeline(BasePipeline):
            def engineer_features(self) -> None:
                pass

            def train_model(self) -> None:
                raise RuntimeError("Training failed: insufficient memory")

        pipeline = FailingPipeline(mock_session, valid_config)
        result = pipeline.execute()

        assert result.status == "failed"
        assert result.error is not None
        assert "Training failed" in result.error
        assert PipelineStage.TRAINING.value in result.error

    def test_stages_completed_tracking(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> None:
        """Test that completed stages are tracked correctly."""

        class PartialPipeline(BasePipeline):
            def engineer_features(self) -> None:
                pass

            def train_model(self) -> None:
                # Fail at training stage
                raise ValueError("Training failed")

        pipeline = PartialPipeline(mock_session, valid_config)
        result = pipeline.execute()

        # Should have completed stages up to (but not including) training
        assert PipelineStage.VALIDATION.value in result.stages_completed
        assert PipelineStage.INFRASTRUCTURE_SETUP.value in result.stages_completed
        assert PipelineStage.INGESTION.value in result.stages_completed
        assert PipelineStage.TRANSFORMATION.value in result.stages_completed
        assert PipelineStage.FEATURE_ENGINEERING.value in result.stages_completed
        assert PipelineStage.TRAINING.value not in result.stages_completed

    def test_abstract_methods_must_be_implemented(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> None:
        """Test that abstract methods must be implemented by subclasses."""

        # This should raise TypeError because abstract methods are not implemented
        with pytest.raises(TypeError):
            BasePipeline(mock_session, valid_config)

    def test_pipeline_stage_enum_values(self) -> None:
        """Test that all pipeline stages are defined correctly."""
        expected_stages = [
            "validation",
            "infrastructure_setup",
            "ingestion",
            "transformation",
            "feature_engineering",
            "training",
            "model_validation",
            "deployment",
            "monitoring",
        ]

        actual_stages = [stage.value for stage in PipelineStage]

        assert len(actual_stages) == len(expected_stages)
        for stage in expected_stages:
            assert stage in actual_stages

    def test_pipeline_emits_stage_metrics_and_events(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> None:
        """Test that pipeline records stage metrics and tracker events."""

        tracker = RecordingTracker()

        class InstrumentedPipeline(BasePipeline):
            def __init__(self, session: Mock, config: PipelineConfig) -> None:
                super().__init__(session, config, tracker=tracker)

            def engineer_features(self) -> None:
                self.logger.info("Engineering features")

            def train_model(self) -> None:
                self.logger.info("Training model")

        pipeline = InstrumentedPipeline(mock_session, valid_config)
        result = pipeline.execute()

        assert result.execution_status == PipelineExecutionStatus.SUCCESS
        assert len(result.stage_metrics) == len(PipelineStage)
        assert tracker.events
        component, event, payload = tracker.events[-1]
        assert component == "InstrumentedPipeline"
        assert event == "pipeline_end"
        assert payload["status"] == PipelineExecutionStatus.SUCCESS.value


class TestPipelineTemplateMethod:
    """Test cases for Template Method pattern implementation."""

    @pytest.fixture
    def mock_session(self) -> Mock:
        """Create a mock Snowflake session."""
        return Mock()

    @pytest.fixture
    def valid_config(self) -> PipelineConfig:
        """Create a valid pipeline configuration."""
        return PipelineConfig(
            name="test_pipeline",
            version="1.0.0",
            environment="dev",
            database="ML_DEV_DB",
            warehouse="ML_TRAINING_WH",
        )

    def test_template_method_calls_all_stages_in_order(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> None:
        """Test that template method calls all stages in correct order."""

        call_order = []

        class OrderTrackingPipeline(BasePipeline):
            def validate_config(self) -> None:
                call_order.append("validate_config")
                super().validate_config()

            def setup_infrastructure(self) -> None:
                call_order.append("setup_infrastructure")
                super().setup_infrastructure()

            def ingest_data(self) -> None:
                call_order.append("ingest_data")
                super().ingest_data()

            def transform_data(self) -> None:
                call_order.append("transform_data")
                super().transform_data()

            def engineer_features(self) -> None:
                call_order.append("engineer_features")

            def train_model(self) -> None:
                call_order.append("train_model")

            def validate_model(self) -> None:
                call_order.append("validate_model")
                super().validate_model()

            def deploy_model(self) -> None:
                call_order.append("deploy_model")
                super().deploy_model()

            def setup_monitoring(self) -> None:
                call_order.append("setup_monitoring")
                super().setup_monitoring()

        pipeline = OrderTrackingPipeline(mock_session, valid_config)
        pipeline.execute()

        expected_order = [
            "validate_config",
            "setup_infrastructure",
            "ingest_data",
            "transform_data",
            "engineer_features",
            "train_model",
            "validate_model",
            "deploy_model",
            "setup_monitoring",
        ]

        assert call_order == expected_order

    def test_subclass_can_override_default_implementations(
        self, mock_session: Mock, valid_config: PipelineConfig
    ) -> None:
        """Test that subclasses can override default implementations."""

        custom_validation_called = False

        class CustomPipeline(BasePipeline):
            def validate_config(self) -> None:
                nonlocal custom_validation_called
                custom_validation_called = True
                # Custom validation logic
                if self.config.name != "test_pipeline":
                    raise ValueError("Invalid pipeline name")

            def engineer_features(self) -> None:
                pass

            def train_model(self) -> None:
                pass

        pipeline = CustomPipeline(mock_session, valid_config)
        result = pipeline.execute()

        assert custom_validation_called
        assert result.status == "success"
