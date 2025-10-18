"""LightGBM training implementation."""

import os
import tempfile
from datetime import datetime
from typing import Any

import joblib

from snowflake_ml_template.core.base.training import (
    BaseTrainer,
    MLFramework,
    TrainingResult,
)
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class LightGBMTrainer(BaseTrainer):
    """LightGBM model trainer."""

    def train(self, data: Any, **kwargs: Any) -> TrainingResult:
        """Train LightGBM model.

        Args:
            data: Input data for training
            **kwargs: Additional keyword arguments

        Returns:
            TrainingResult: Result of the training operation
        """
        start_time = datetime.utcnow()

        try:
            import lightgbm as lgb

            # Convert Snowpark DataFrame to pandas if needed
            if hasattr(data, "to_pandas"):
                data = data.to_pandas()

            # Extract features and target
            # Exclude target and sample_weight from features
            exclude_cols = [self.config.target_column]
            if "SAMPLE_WEIGHT" in data.columns:
                exclude_cols.append("SAMPLE_WEIGHT")

            X = data.drop(columns=exclude_cols)
            y = data[self.config.target_column]

            # Create LightGBM model
            model = lgb.LGBMClassifier(**self.config.model_config.hyperparameters)

            # Train
            model.fit(X, y)

            # Save model to local temp file
            temp_dir = tempfile.gettempdir()
            model_filename = (
                f"lightgbm_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            model_path = os.path.join(temp_dir, model_filename)
            self.save_model(model, model_path)

            return TrainingResult(
                status="success",
                strategy=self.config.strategy,
                framework=MLFramework.LIGHTGBM,
                model_artifact_path=model_path,
                start_time=start_time,
                end_time=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                status="failed",
                strategy=self.config.strategy,
                framework=MLFramework.LIGHTGBM,
                start_time=start_time,
                end_time=datetime.utcnow(),
                error=str(e),
            )

    def validate(self) -> bool:
        """Validate the training configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return bool(self.config.training_table and self.config.warehouse)

    def save_model(self, model: Any, path: str) -> str:
        """Save the trained model to disk.

        Args:
            model: Trained model object
            path: Path to save the model

        Returns:
            str: Path where model was saved
        """
        joblib.dump(model, path)
        return path

    def load_model(self, path: str) -> Any:
        """Load a trained model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Any: Loaded model object
        """
        return joblib.load(path)
