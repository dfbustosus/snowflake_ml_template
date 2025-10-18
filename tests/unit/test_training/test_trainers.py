"""Tests for ML framework trainers."""

import pytest

from snowflake_ml_template.core.base.training import (
    BaseModelConfig,
    MLFramework,
    TrainingConfig,
    TrainingStrategy,
)
from snowflake_ml_template.training.frameworks.lightgbm_trainer import LightGBMTrainer
from snowflake_ml_template.training.frameworks.sklearn_trainer import SklearnTrainer
from snowflake_ml_template.training.frameworks.xgboost_trainer import XGBoostTrainer


@pytest.fixture
def training_config():
    """Create training configuration."""
    return TrainingConfig(
        strategy=TrainingStrategy.SINGLE_NODE,
        model_config=BaseModelConfig(
            framework=MLFramework.XGBOOST,
            model_type="classifier",
            hyperparameters={"max_depth": 3},
        ),
        training_database="TEST_DB",
        training_schema="FEATURES",
        training_table="TRAINING_DATA",
        warehouse="TEST_WH",
        target_column="TARGET",
    )


def test_xgboost_trainer_validation(training_config):
    """Test XGBoost trainer validation."""
    trainer = XGBoostTrainer(training_config)
    assert trainer.validate() is True


def test_sklearn_trainer_validation(training_config):
    """Test sklearn trainer validation."""
    training_config.model_config.framework = MLFramework.SKLEARN
    trainer = SklearnTrainer(training_config)
    assert trainer.validate() is True


def test_lightgbm_trainer_validation(training_config):
    """Test LightGBM trainer validation."""
    training_config.model_config.framework = MLFramework.LIGHTGBM
    trainer = LightGBMTrainer(training_config)
    assert trainer.validate() is True


def test_trainer_validation_fails_without_table(training_config):
    """Test trainer validation fails without training table."""
    training_config.training_table = None
    trainer = XGBoostTrainer(training_config)
    assert trainer.validate() is False
