"""Integration tests for multi-framework training."""

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
from snowflake_ml_template.training.orchestrator import TrainingOrchestrator


@pytest.fixture
def training_orchestrator(mock_session):
    """Create training orchestrator."""
    return TrainingOrchestrator(mock_session)


def test_xgboost_framework_integration(mock_session, training_orchestrator):
    """Test XGBoost framework integration."""
    config = TrainingConfig(
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

    trainer = XGBoostTrainer(config)
    training_orchestrator.register_trainer("xgboost", trainer)

    assert "xgboost" in training_orchestrator.trainers
    assert training_orchestrator.trainers["xgboost"].validate()


def test_sklearn_framework_integration(mock_session, training_orchestrator):
    """Test Scikit-learn framework integration."""
    config = TrainingConfig(
        strategy=TrainingStrategy.SINGLE_NODE,
        model_config=BaseModelConfig(
            framework=MLFramework.SKLEARN,
            model_type="random_forest",
            hyperparameters={"n_estimators": 10},
        ),
        training_database="TEST_DB",
        training_schema="FEATURES",
        training_table="TRAINING_DATA",
        warehouse="TEST_WH",
        target_column="TARGET",
    )

    trainer = SklearnTrainer(config)
    training_orchestrator.register_trainer("sklearn", trainer)

    assert "sklearn" in training_orchestrator.trainers
    assert training_orchestrator.trainers["sklearn"].validate()


def test_lightgbm_framework_integration(mock_session, training_orchestrator):
    """Test LightGBM framework integration."""
    config = TrainingConfig(
        strategy=TrainingStrategy.SINGLE_NODE,
        model_config=BaseModelConfig(
            framework=MLFramework.LIGHTGBM,
            model_type="classifier",
            hyperparameters={"num_leaves": 31},
        ),
        training_database="TEST_DB",
        training_schema="FEATURES",
        training_table="TRAINING_DATA",
        warehouse="TEST_WH",
        target_column="TARGET",
    )

    trainer = LightGBMTrainer(config)
    training_orchestrator.register_trainer("lightgbm", trainer)

    assert "lightgbm" in training_orchestrator.trainers
    assert training_orchestrator.trainers["lightgbm"].validate()


def test_multiple_frameworks_registered(mock_session, training_orchestrator):
    """Test multiple frameworks can be registered simultaneously."""
    configs = {
        "xgboost": TrainingConfig(
            strategy=TrainingStrategy.SINGLE_NODE,
            model_config=BaseModelConfig(
                framework=MLFramework.XGBOOST,
                model_type="classifier",
                hyperparameters={},
            ),
            training_database="TEST_DB",
            training_schema="FEATURES",
            training_table="TRAINING_DATA",
            warehouse="TEST_WH",
            target_column="TARGET",
        ),
        "sklearn": TrainingConfig(
            strategy=TrainingStrategy.SINGLE_NODE,
            model_config=BaseModelConfig(
                framework=MLFramework.SKLEARN,
                model_type="random_forest",
                hyperparameters={},
            ),
            training_database="TEST_DB",
            training_schema="FEATURES",
            training_table="TRAINING_DATA",
            warehouse="TEST_WH",
            target_column="TARGET",
        ),
        "lightgbm": TrainingConfig(
            strategy=TrainingStrategy.SINGLE_NODE,
            model_config=BaseModelConfig(
                framework=MLFramework.LIGHTGBM,
                model_type="classifier",
                hyperparameters={},
            ),
            training_database="TEST_DB",
            training_schema="FEATURES",
            training_table="TRAINING_DATA",
            warehouse="TEST_WH",
            target_column="TARGET",
        ),
    }

    training_orchestrator.register_trainer(
        "xgboost", XGBoostTrainer(configs["xgboost"])
    )
    training_orchestrator.register_trainer(
        "sklearn", SklearnTrainer(configs["sklearn"])
    )
    training_orchestrator.register_trainer(
        "lightgbm", LightGBMTrainer(configs["lightgbm"])
    )

    assert len(training_orchestrator.trainers) == 3
    assert all(
        name in training_orchestrator.trainers
        for name in ["xgboost", "sklearn", "lightgbm"]
    )
