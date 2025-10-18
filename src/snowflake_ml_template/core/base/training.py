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
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


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

    status: str
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
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            self.duration_seconds = delta.total_seconds()


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

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the trainer.

        Args:
            config: Training configuration

        Raises:
            ValueError: If config is None
        """
        if config is None:
            raise ValueError("Config cannot be None")

        self.config = config
        self.logger = self._get_logger()

    def _get_logger(self) -> Any:
        """Get logger instance.

        This is a placeholder that will be replaced with proper
        structured logging in Day 3.

        Returns:
            Logger instance
        """
        import logging

        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

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
