"""Base pipeline abstraction implementing the Template Method pattern.

This module defines the core pipeline abstraction that all ML pipelines must
inherit from. It implements the Template Method design pattern, where the
overall pipeline structure is defined in the base class, but specific steps
are implemented by subclasses.

The pipeline follows these stages:
1. Configuration validation
2. Infrastructure setup
3. Data ingestion
4. Data transformation
5. Feature engineering (abstract - model-specific)
6. Model training (abstract - model-specific)
7. Model validation
8. Model deployment
9. Monitoring setup

Classes:
    PipelineStage: Enum defining pipeline execution stages
    PipelineConfig: Configuration for pipeline execution
    PipelineResult: Result of pipeline execution
    BasePipeline: Abstract base class for all ML pipelines
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.tracking import (
    ExecutionEventTracker,
    emit_tracker_event,
)
from snowflake_ml_template.utils.logging import StructuredLogger, get_logger


class PipelineStage(Enum):
    """Enumeration of pipeline execution stages.

    Each stage represents a distinct phase in the ML pipeline lifecycle.
    Stages are executed sequentially in the order defined here.
    """

    VALIDATION = "validation"
    INFRASTRUCTURE_SETUP = "infrastructure_setup"
    INGESTION = "ingestion"
    TRANSFORMATION = "transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    MODEL_VALIDATION = "model_validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


class PipelineExecutionStatus(Enum):
    """Enumeration of pipeline execution outcomes."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    This dataclass encapsulates all configuration needed to execute a pipeline.
    It follows the principle of explicit configuration over implicit behavior.

    Attributes:
        name: Unique identifier for the pipeline
        version: Semantic version of the pipeline
        environment: Target environment (dev, test, prod)
        database: Target Snowflake database
        warehouse: Snowflake warehouse for execution
        ingestion: Configuration for data ingestion
        transformation: Configuration for data transformation
        features: Configuration for feature engineering
        training: Configuration for model training
        deployment: Configuration for model deployment
        monitoring: Configuration for monitoring
        metadata: Additional metadata as key-value pairs
    """

    name: str
    version: str
    environment: str
    database: str
    warehouse: str
    ingestion: Optional[Dict[str, Any]] = None
    transformation: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None
    deployment: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")
        if not self.version:
            raise ValueError("Pipeline version cannot be empty")
        if self.environment not in ["dev", "test", "prod"]:
            raise ValueError(
                f"Invalid environment: {self.environment}. "
                "Must be one of: dev, test, prod"
            )


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    This dataclass captures the outcome of a pipeline run, including
    status, timing information, and any errors encountered.

    Attributes:
        status: Execution status (success, failed, partial)
        pipeline_name: Name of the executed pipeline
        pipeline_version: Version of the executed pipeline
        start_time: When execution started
        end_time: When execution completed
        duration_seconds: Total execution time in seconds
        stages_completed: List of successfully completed stages
        error: Error message if execution failed
        metadata: Additional result metadata
    """

    status: str | PipelineExecutionStatus
    pipeline_name: str = ""
    pipeline_version: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate duration if start and end times are provided."""
        if isinstance(self.status, PipelineExecutionStatus):
            self.status = self.status.value

        if self.start_time and self.end_time:
            start = self.start_time
            end = self.end_time

            if (start.tzinfo is None) != (end.tzinfo is None):
                if start.tzinfo is not None:
                    start = start.replace(tzinfo=None)
                if end.tzinfo is not None:
                    end = end.replace(tzinfo=None)

            delta = end - start
            self.duration_seconds = delta.total_seconds()

    @property
    def execution_status(self) -> PipelineExecutionStatus:
        """Return the execution status as an enum."""
        if isinstance(self.status, PipelineExecutionStatus):
            return self.status

        try:
            return PipelineExecutionStatus(self.status)
        except ValueError:
            if self.error:
                return PipelineExecutionStatus.FAILED
            if self.stages_completed:
                return PipelineExecutionStatus.PARTIAL
            return PipelineExecutionStatus.SUCCESS

    def add_stage_metric(self, stage: str, metrics: Dict[str, float]) -> None:
        """Store metrics associated with a stage."""
        self.stage_metrics[stage] = metrics


class BasePipeline(ABC):
    """Abstract base class for all ML pipelines.

    This class implements the Template Method design pattern. The execute()
    method defines the overall pipeline structure, calling specific methods
    in a defined order. Subclasses must implement abstract methods for
    model-specific logic while inheriting common functionality.

    The pipeline ensures:
    - Consistent execution flow across all models
    - Proper error handling and logging
    - Resource cleanup
    - Idempotent operations where possible

    Attributes:
        session: Snowflake session for database operations
        config: Pipeline configuration
        logger: Logger instance for structured logging
        _current_stage: Currently executing pipeline stage
        _start_time: Pipeline execution start time

    Example:
        >>> class FraudDetectionPipeline(BasePipeline):
        ...     def engineer_features(self) -> None:
        ...         # Implement fraud-specific features
        ...         pass
        ...
        ...     def train_model(self) -> None:
        ...         # Implement fraud model training
        ...         pass
        >>>
        >>> pipeline = FraudDetectionPipeline(session, config)
        >>> result = pipeline.execute()
    """

    def __init__(
        self,
        session: Session,
        config: PipelineConfig,
        tracker: ExecutionEventTracker | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            session: Active Snowflake session
            config: Pipeline configuration
            tracker: Optional execution tracker for telemetry events

        Raises:
            ValueError: If session or config is None
        """
        if session is None:
            raise ValueError("Session cannot be None")
        if config is None:
            raise ValueError("Config cannot be None")

        self.session = session
        self.config = config
        self._current_stage: Optional[PipelineStage] = None
        self._start_time: Optional[datetime] = None
        self._stages_completed: List[str] = []
        self._stage_metrics: Dict[str, Dict[str, float]] = {}
        self._tracker = tracker

        self.logger: StructuredLogger = self._get_logger()

    def _get_logger(self) -> StructuredLogger:
        """Retrieve structured logger scoped to the pipeline."""
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(logger_name)

    @property
    def tracker(self) -> ExecutionEventTracker | None:
        """Return the configured execution tracker."""
        return self._tracker

    def _emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit a tracker event with standard metadata."""
        base_payload: Dict[str, Any] = {
            "pipeline": self.config.name,
            "version": self.config.version,
            "stage": self._current_stage.value if self._current_stage else None,
        }
        base_payload.update(payload)
        emit_tracker_event(
            tracker=self._tracker,
            component=self.__class__.__name__,
            event=event,
            payload=base_payload,
        )

    def _on_stage_start(self, stage: PipelineStage) -> None:
        """Execute hook before a stage starts."""
        pass

    def _on_stage_end(self, stage: PipelineStage, duration_seconds: float) -> None:
        """Execute hook after a stage completes successfully."""
        pass

    def _on_stage_error(self, stage: PipelineStage, error: Exception) -> None:
        """Handle stage failure inside the hook."""
        pass

    def execute(self) -> PipelineResult:
        """Run the entire ML pipeline.

        The method orchestrates the following stages sequentially:
            1. Validation.
            2. Infrastructure setup.
            3. Data ingestion.
            4. Data transformation.
            5. Feature engineering.
            6. Model training.
            7. Model validation.
            8. Model deployment.
            9. Monitoring setup.
        """
        self._start_time = datetime.now(timezone.utc)
        self._stage_metrics = {}
        self._emit_event(
            event="pipeline_start",
            payload={"timestamp": self._start_time.isoformat()},
        )
        self.logger.info(
            f"Starting pipeline execution: {self.config.name} "
            f"v{self.config.version}"
        )

        try:
            # Stage 1: Validation
            self._execute_stage(PipelineStage.VALIDATION, self.validate_config)

            # Stage 2: Infrastructure Setup
            self._execute_stage(
                PipelineStage.INFRASTRUCTURE_SETUP, self.setup_infrastructure
            )

            # Stage 3: Data Ingestion
            self._execute_stage(PipelineStage.INGESTION, self.ingest_data)

            # Stage 4: Data Transformation
            self._execute_stage(PipelineStage.TRANSFORMATION, self.transform_data)

            # Stage 5: Feature Engineering (abstract - model-specific)
            self._execute_stage(
                PipelineStage.FEATURE_ENGINEERING, self.engineer_features
            )

            # Stage 6: Model Training (abstract - model-specific)
            self._execute_stage(PipelineStage.TRAINING, self.train_model)

            # Stage 7: Model Validation
            self._execute_stage(PipelineStage.MODEL_VALIDATION, self.validate_model)

            # Stage 8: Model Deployment
            self._execute_stage(PipelineStage.DEPLOYMENT, self.deploy_model)

            # Stage 9: Monitoring Setup
            self._execute_stage(PipelineStage.MONITORING, self.setup_monitoring)

            # Success
            end_time = datetime.now(timezone.utc)
            self.logger.info(
                f"Pipeline execution completed successfully: "
                f"{self.config.name} v{self.config.version}"
            )
            self._emit_event(
                event="pipeline_end",
                payload={
                    "timestamp": end_time.isoformat(),
                    "status": PipelineExecutionStatus.SUCCESS.value,
                    "stage_metrics": self._stage_metrics,
                },
            )

            return PipelineResult(
                status=PipelineExecutionStatus.SUCCESS,
                pipeline_name=self.config.name,
                pipeline_version=self.config.version,
                start_time=self._start_time,
                end_time=end_time,
                stages_completed=self._stages_completed,
                stage_metrics=self._stage_metrics.copy(),
            )

        except Exception as e:
            # Failure
            end_time = datetime.now(timezone.utc)
            stage_name = self._current_stage.value if self._current_stage else "unknown"
            error_msg = f"Pipeline execution failed at stage {stage_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._emit_event(
                event="pipeline_end",
                payload={
                    "timestamp": end_time.isoformat(),
                    "status": PipelineExecutionStatus.FAILED.value,
                    "error": error_msg,
                    "stage_metrics": self._stage_metrics,
                },
            )

            return PipelineResult(
                status=PipelineExecutionStatus.FAILED,
                pipeline_name=self.config.name,
                pipeline_version=self.config.version,
                start_time=self._start_time,
                end_time=end_time,
                stages_completed=self._stages_completed,
                error=error_msg,
                stage_metrics=self._stage_metrics.copy(),
            )

    def _execute_stage(self, stage: PipelineStage, stage_func: Any) -> None:
        """Execute a single pipeline stage with error handling."""
        self._current_stage = stage
        self.logger.info(f"Executing stage: {stage.value}")
        self._emit_event(
            event="stage_start",
            payload={
                "stage": stage.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._on_stage_start(stage)
        stage_start = datetime.now(timezone.utc)

        try:
            stage_func()
            duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
            self._stages_completed.append(stage.value)
            self._stage_metrics[stage.value] = {"duration_seconds": duration}
            self.logger.info(f"Stage completed: {stage.value}")
            self._emit_event(
                event="stage_end",
                payload={
                    "stage": stage.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": self._stage_metrics[stage.value],
                },
            )
            self._on_stage_end(stage, duration)
        except Exception as e:
            duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
            self.logger.error(f"Stage failed: {stage.value} - {str(e)}", exc_info=True)
            self._stage_metrics[stage.value] = {"duration_seconds": duration}
            self._emit_event(
                event="stage_error",
                payload={
                    "stage": stage.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "metrics": self._stage_metrics[stage.value],
                },
            )
            self._on_stage_error(stage, e)
            raise

    # =========================================================================
    # Common implementations (can be overridden by subclasses)
    # =========================================================================

    def validate_config(self) -> None:
        """Validate pipeline configuration.

        This method performs basic validation of the pipeline configuration.
        Subclasses can override to add model-specific validation.

        Raises:
            ValueError: If configuration is invalid
        """
        self.logger.info("Validating pipeline configuration")

        # Basic validation is done in PipelineConfig.__post_init__
        # Subclasses can add additional validation here

        self.logger.info("Configuration validation passed")

    def setup_infrastructure(self) -> None:
        """Set up required Snowflake infrastructure.

        This method ensures all required databases, schemas, and other
        infrastructure components exist. It uses the infrastructure module
        which will be implemented in Day 4-5.

        Default implementation is a no-op. Subclasses can override to
        set up model-specific infrastructure.
        """
        self.logger.info("Setting up infrastructure")
        # Infrastructure setup will be implemented in Day 4-5
        self.logger.info("Infrastructure setup completed")

    def ingest_data(self) -> None:
        """Ingest data from source systems.

        This method uses the ingestion engine to load data from various
        sources. The ingestion engine will be implemented in Phase 3.

        Default implementation is a no-op. Subclasses can override to
        implement model-specific ingestion logic.
        """
        self.logger.info("Ingesting data")
        # Ingestion engine will be implemented in Phase 3
        self.logger.info("Data ingestion completed")

    def transform_data(self) -> None:
        """Transform raw data into analysis-ready format.

        This method uses the transformation engine to process raw data.
        The transformation engine will be implemented in Phase 3.

        Default implementation is a no-op. Subclasses can override to
        implement model-specific transformations.
        """
        self.logger.info("Transforming data")
        # Transformation engine will be implemented in Phase 3
        self.logger.info("Data transformation completed")

    def validate_model(self) -> None:
        """Validate trained model.

        This method performs validation of the trained model, including
        performance metrics, fairness auditing, and robustness testing.
        The validation framework will be implemented in Phase 5.

        Default implementation is a no-op. Subclasses can override to
        implement model-specific validation.
        """
        self.logger.info("Validating model")
        # Validation framework will be implemented in Phase 5
        self.logger.info("Model validation completed")

    def deploy_model(self) -> None:
        """Deploy trained model.

        This method uses the deployment engine to deploy the model.
        The deployment engine will be implemented in Phase 3.

        Default implementation is a no-op. Subclasses can override to
        implement model-specific deployment logic.
        """
        self.logger.info("Deploying model")
        # Deployment engine will be implemented in Phase 3
        self.logger.info("Model deployment completed")

    def setup_monitoring(self) -> None:
        """Set up monitoring for deployed model.

        This method configures monitoring for the deployed model, including
        performance tracking, drift detection, and alerting.
        The monitoring system will be implemented in Phase 5.

        Default implementation is a no-op. Subclasses can override to
        implement model-specific monitoring.
        """
        self.logger.info("Setting up monitoring")
        # Monitoring system will be implemented in Phase 5
        self.logger.info("Monitoring setup completed")

    # =========================================================================
    # Abstract methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def engineer_features(self) -> None:
        """Engineer features for the model.

        This is an abstract method that must be implemented by all subclasses.
        It defines the model-specific feature engineering logic.

        Subclasses should:
        1. Define entities and feature views
        2. Register them with the feature store
        3. Generate training datasets

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement engineer_features()"
        )

    @abstractmethod
    def train_model(self) -> None:
        """Train the ML model.

        This is an abstract method that must be implemented by all subclasses.
        It defines the model-specific training logic.

        Subclasses should:
        1. Load training data from feature store
        2. Configure training parameters
        3. Train the model
        4. Save artifacts to model registry

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement train_model()"
        )
