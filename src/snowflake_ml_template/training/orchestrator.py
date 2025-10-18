"""Training orchestrator for managing training operations."""

from typing import Any, Dict

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.training import BaseTrainer, TrainingResult
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingOrchestrator:
    """Orchestrate model training operations.

    This class manages the registration and execution of different model trainers.
    It provides a unified interface to train models using different frameworks.

    Example:
        >>> orchestrator = TrainingOrchestrator(session)
        >>> orchestrator.register_trainer("xgboost", XGBoostTrainer(config))
        >>> result = orchestrator.execute("xgboost", training_data)
    """

    def __init__(self, session: Session) -> None:
        """Initialize the training orchestrator.

        Args:
            session: Snowflake session for database operations
        """
        self.session = session
        self.trainers: Dict[str, BaseTrainer] = {}
        self.logger = get_logger(__name__)

    def register_trainer(self, name: str, trainer: BaseTrainer) -> None:
        """Register a trainer with the orchestrator.

        Args:
            name: Name to register the trainer under
            trainer: Trainer instance to register
        """
        self.trainers[name] = trainer
        self.logger.info(f"Registered trainer: {name}")

    def execute(self, trainer_name: str, data: Any) -> TrainingResult:
        """Execute training using the specified trainer.

        Args:
            trainer_name: Name of the registered trainer to use
            data: Training data

        Returns:
            TrainingResult: Result of the training operation

        Raises:
            ValueError: If trainer is not found or validation fails
        """
        if trainer_name not in self.trainers:
            raise ValueError(f"Trainer not found: {trainer_name}")

        trainer = self.trainers[trainer_name]

        if not trainer.validate():
            raise ValueError(f"Trainer validation failed: {trainer_name}")

        result = trainer.train(data)

        self.logger.info(
            f"Training completed: {trainer_name}", extra={"status": result.status}
        )

        return result
