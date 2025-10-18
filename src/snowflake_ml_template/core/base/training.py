"""Base abstractions for model training strategies.

This module defines the interface for model training. Different training
strategies (single-node, distributed, GPU, MMT) and frameworks (sklearn,
XGBoost, PyTorch) implement the same interface.

Classes:
    TrainingStrategy: Enum of training strategies
    MLFramework: Enum of ML frameworks
    BaseModelConfig: Base configuration for models
    TrainingConfig: Configuration for training operation
    TrainingResult: Result of training operation
    BaseTrainer: Abstract base class for model trainers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from snowflake_ml_template.core.base.tracking import (
    ExecutionEventTracker,
    emit_tracker_event,
)
from snowflake_ml_template.utils.logging import StructuredLogger, get_logger


class TrainingStrategy(Enum):
    """Enumeration of supported training strategies."""

    SINGLE_NODE = "single_node"
    DISTRIBUTED = "distributed"
    GPU = "gpu"
    MANY_MODEL = "many_model"


class MLFramework(Enum):
    """Enumeration of supported ML frameworks."""

    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class TrainingStatus(Enum):
    """Enumeration of training outcomes."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BaseModelConfig:
    """Base configuration for ML models.

    Attributes:
        framework: ML framework to use
        model_type: Type of model (classifier, regressor, etc.)
        hyperparameters: Model hyperparameters
        random_state: Random seed for reproducibility
        metadata: Additional model-specific metadata
    """

    framework: MLFramework
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model configuration."""
        if not self.model_type:
            raise ValueError("Model type cannot be empty")


@dataclass
class TrainingConfig:
    """Configuration for model training operation.

    Attributes:
        strategy: Training strategy to use
        model_config: Model configuration
        training_database: Database containing training data
        training_schema: Schema containing training data
        training_table: Training data table name
        warehouse: Snowflake warehouse for training
        validation_split: Fraction of data for validation (0.0-1.0)
        test_split: Fraction of data for testing (0.0-1.0)
        target_column: Name of target column
        feature_columns: List of feature column names
        cv_folds: Number of cross-validation folds (0 for no CV)
        early_stopping: Whether to use early stopping
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        metadata: Additional training-specific metadata
    """

    strategy: TrainingStrategy
    model_config: BaseModelConfig
    training_database: str
    training_schema: str
    training_table: str
    warehouse: str
    validation_split: float = 0.2
    test_split: float = 0.1
    target_column: str = "target"
    feature_columns: Optional[list[str]] = None
    cv_folds: int = 0
    early_stopping: bool = False
    max_epochs: int = 100
    batch_size: int = 32
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if not self.training_database:
            raise ValueError("Training database cannot be empty")
        if not self.warehouse:
            raise ValueError("Warehouse cannot be empty")

        if not (0.0 <= self.validation_split < 1.0):
            raise ValueError(
                f"Validation split must be between 0.0 and 1.0, "
                f"got {self.validation_split}"
            )

        if not (0.0 <= self.test_split < 1.0):
            raise ValueError(
                f"Test split must be between 0.0 and 1.0, " f"got {self.test_split}"
            )

        if self.validation_split + self.test_split >= 1.0:
            raise ValueError(
                f"Sum of validation_split ({self.validation_split}) and "
                f"test_split ({self.test_split}) must be less than 1.0"
            )


@dataclass
class TrainingResult:
    """Result of model training operation.

    Attributes:
        status: Training status (success, failed)
        strategy: Training strategy used
        framework: ML framework used
        model_artifact_path: Path to saved model artifact
        metrics: Training and validation metrics
        best_epoch: Best epoch number (for early stopping)
        total_epochs: Total number of epochs trained
        training_samples: Number of training samples
        validation_samples: Number of validation samples
        test_samples: Number of test samples
        start_time: When training started
        end_time: When training completed
        duration_seconds: Total training time in seconds
        error: Error message if training failed
        metadata: Additional result metadata
    """

    status: str | TrainingStatus
    strategy: TrainingStrategy
    framework: MLFramework
    model_artifact_path: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    best_epoch: int = 0
    total_epochs: int = 0
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate duration if start and end times are provided."""
        if isinstance(self.status, TrainingStatus):
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
    def training_status(self) -> TrainingStatus:
        """Return the training status as an enum."""
        if isinstance(self.status, TrainingStatus):
            return self.status

        try:
            return TrainingStatus(self.status)
        except ValueError:
            if self.error:
                return TrainingStatus.FAILED
            if self.metrics:
                return TrainingStatus.PARTIAL
            return TrainingStatus.SUCCESS


class BaseTrainer(ABC):
    """Abstract base class for model trainers.

    This class defines the interface that all training implementations
    must follow. It supports different training strategies and frameworks
    through a common interface.

    Attributes:
        config: Training configuration
        logger: Logger instance for structured logging

    Example:
        >>> class XGBoostTrainer(BaseTrainer):
        ...     def train(self, data, **kwargs) -> TrainingResult:
        ...         # Implement XGBoost training
        ...         pass
        >>>
        >>> trainer = XGBoostTrainer(config)
        >>> result = trainer.train(training_data)
    """

    def __init__(
        self,
        config: TrainingConfig,
        tracker: ExecutionEventTracker | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Training configuration
            tracker: Optional execution tracker for telemetry events

        Raises:
            ValueError: If config is None
        """
        if config is None:
            raise ValueError("Config cannot be None")

        self.config = config
        self.logger: StructuredLogger = self._get_logger()
        self._tracker = tracker

    def _get_logger(self) -> StructuredLogger:
        """Return a structured logger scoped to the trainer."""
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(logger_name)

    @property
    def tracker(self) -> ExecutionEventTracker | None:
        """Return the configured execution tracker."""
        return self._tracker

    def _emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit a training event."""
        base_payload: Dict[str, Any] = {
            "strategy": self.config.strategy.value,
            "framework": self.config.model_config.framework.value,
            "training_table": self.get_training_table_name(),
        }
        base_payload.update(payload)
        emit_tracker_event(
            tracker=self._tracker,
            component=self.__class__.__name__,
            event=event,
            payload=base_payload,
        )

    @abstractmethod
    def train(self, data: Any, **kwargs: Any) -> TrainingResult:
        """Train the model.

        This is the main method that performs model training. Subclasses
        must implement this method with their specific training logic.

        Args:
            data: Training data (DataFrame, numpy array, etc.)
            **kwargs: Additional training-specific parameters

        Returns:
            TrainingResult with status, metrics, and model artifact path

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement train()")

    @abstractmethod
    def validate(self) -> bool:
        """Validate training configuration.

        This method validates that the training configuration is correct
        and all required resources are available.

        Returns:
            True if configuration is valid, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )

    def pre_train(self, data: Any, **kwargs: Any) -> None:
        """Execute hook before training begins."""
        pass

    def post_train(self, result: TrainingResult) -> None:
        """Execute hook after successful training."""
        pass

    def on_train_error(self, error: Exception) -> None:
        """Handle training failure in hook."""
        pass

    def pre_data_validation(self, data: Any, **kwargs: Any) -> None:
        """Perform governance checks before training commences."""
        pass

    def validate_training_data(self, data: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return validation report for training data."""
        return {}

    def post_data_validation(self, report: Dict[str, Any]) -> None:
        """Handle governance reporting after data validation."""
        pass

    def on_data_validation_error(self, error: Exception) -> None:
        """React to training data governance failures."""
        pass

    def post_model_governance(
        self, result: TrainingResult, report: Dict[str, Any]
    ) -> None:
        """Handle governance activities after training completes."""
        pass

    @abstractmethod
    def save_model(self, model: Any, path: str) -> str:
        """Save trained model to specified path.

        Args:
            model: Trained model object
            path: Path to save model (Snowflake stage or local)

        Returns:
            Full path to saved model

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement save_model()"
        )

    @abstractmethod
    def load_model(self, path: str) -> Any:
        """Load model from specified path.

        Args:
            path: Path to load model from

        Returns:
            Loaded model object

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement load_model()"
        )

    def get_training_table_name(self) -> str:
        """Get fully qualified training table name.

        Returns:
            Fully qualified table name (DATABASE.SCHEMA.TABLE)
        """
        return (
            f"{self.config.training_database}."
            f"{self.config.training_schema}."
            f"{self.config.training_table}"
        )

    def execute_training(self, data: Any, **kwargs: Any) -> TrainingResult:
        """Execute training with lifecycle hooks and tracking."""
        try:
            self.pre_data_validation(data, **kwargs)
            validation_report = self.validate_training_data(data, **kwargs)
            self.post_data_validation(validation_report)
        except Exception as validation_error:
            self.on_data_validation_error(validation_error)
            raise

        self._emit_event(
            event="training_start",
            payload={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_summary": kwargs.get("data_summary"),
            },
        )
        self.pre_train(data, **kwargs)
        start_time = datetime.now(timezone.utc)

        try:
            result = self.train(data=data, **kwargs)
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.metrics.setdefault("duration_seconds", duration)
            result.start_time = result.start_time or start_time
            result.end_time = result.end_time or datetime.now(timezone.utc)
            self.post_model_governance(result, validation_report)
            self.post_train(result)
            self._emit_event(
                event="training_end",
                payload={
                    "status": result.status,
                    "metrics": result.metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return result
        except Exception as exc:
            self.logger.error("Training failed", exc_info=True)
            self.on_train_error(exc)
            self._emit_event(
                event="training_error",
                payload={
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
