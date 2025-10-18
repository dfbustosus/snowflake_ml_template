"""Tests for base training."""

import pytest

from snowflake_ml_template.core.base.training import (
    BaseModelConfig,
    BaseTrainer,
    MLFramework,
    TrainingConfig,
    TrainingStrategy,
)


def test_base_model_config_validation():
    """Test base model config validation."""
    with pytest.raises(ValueError):
        BaseModelConfig(framework=MLFramework.SKLEARN, model_type="")
    cfg = BaseModelConfig(framework=MLFramework.SKLEARN, model_type="classifier")
    assert cfg.framework == MLFramework.SKLEARN


def test_training_config_validation_edges():
    """Test training config validation edges."""
    model_cfg = BaseModelConfig(framework=MLFramework.SKLEARN, model_type="clf")
    # missing database
    with pytest.raises(ValueError):
        TrainingConfig(
            strategy=TrainingStrategy.SINGLE_NODE,
            model_config=model_cfg,
            training_database="",
            training_schema="SC",
            training_table="T",
            warehouse="WH",
            validation_split=0.2,
            test_split=0.1,
        )
    # missing warehouse
    with pytest.raises(ValueError):
        TrainingConfig(
            strategy=TrainingStrategy.SINGLE_NODE,
            model_config=model_cfg,
            training_database="DB",
            training_schema="SC",
            training_table="T",
            warehouse="",
            validation_split=0.2,
            test_split=0.1,
        )
    # invalid splits
    with pytest.raises(ValueError):
        TrainingConfig(
            strategy=TrainingStrategy.SINGLE_NODE,
            model_config=model_cfg,
            training_database="DB",
            training_schema="SC",
            training_table="T",
            warehouse="WH",
            validation_split=1.0,
            test_split=0.0,
        )
    with pytest.raises(ValueError):
        TrainingConfig(
            strategy=TrainingStrategy.SINGLE_NODE,
            model_config=model_cfg,
            training_database="DB",
            training_schema="SC",
            training_table="T",
            warehouse="WH",
            validation_split=0.6,
            test_split=0.5,
        )
    # ok
    ok = TrainingConfig(
        strategy=TrainingStrategy.SINGLE_NODE,
        model_config=model_cfg,
        training_database="DB",
        training_schema="SC",
        training_table="T",
        warehouse="WH",
        validation_split=0.2,
        test_split=0.1,
    )
    assert ok.strategy == TrainingStrategy.SINGLE_NODE


class DummyTrainer(BaseTrainer):
    """Dummy trainer for testing."""

    def train(self, data, **kwargs):  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def validate(self) -> bool:  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def save_model(self, model, path):  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def load_model(self, path):  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError


def test_base_trainer_helpers():
    """Test base trainer helpers."""
    model_cfg = BaseModelConfig(framework=MLFramework.SKLEARN, model_type="clf")
    cfg = TrainingConfig(
        strategy=TrainingStrategy.SINGLE_NODE,
        model_config=model_cfg,
        training_database="DB",
        training_schema="SC",
        training_table="T",
        warehouse="WH",
    )
    trainer = DummyTrainer(cfg)
    assert trainer.get_training_table_name() == "DB.SC.T"
