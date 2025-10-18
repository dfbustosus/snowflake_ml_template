"""Tests for base training."""

import pytest

from snowflake_ml_template.core.base.training import (
    BaseModelConfig,
    BaseTrainer,
    MLFramework,
    TrainingConfig,
    TrainingResult,
    TrainingStatus,
    TrainingStrategy,
)


class RecordingTracker:
    """Simple tracker implementation for training tests."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.events = []

    def record_event(self, component: str, event: str, payload: dict) -> None:
        """Record an event."""
        self.events.append((component, event, payload))


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

    def __init__(self, config: TrainingConfig, tracker=None) -> None:
        """Initialize the trainer."""
        super().__init__(config, tracker=tracker)
        self.pre_called = False
        self.post_called = False
        self.error_called = False
        self.data_validation_pre_called = False
        self.data_validation_report: dict | None = None
        self.data_validation_failure_called = False
        self.model_governance_called = False
        self.raise_data_validation_error = False

    def train(self, data, **kwargs):  # pragma: no cover
        """Return a successful training result."""
        return TrainingResult(
            status=TrainingStatus.SUCCESS,
            strategy=self.config.strategy,
            framework=self.config.model_config.framework,
            metrics={"accuracy": 0.9},
        )

    def validate(self) -> bool:  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def save_model(self, model, path):  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def load_model(self, path):  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def pre_data_validation(self, data, **kwargs):
        """Pre-data validation hook."""
        self.data_validation_pre_called = True

    def validate_training_data(self, data, **kwargs):
        """Validate training data hook."""
        if self.raise_data_validation_error:
            raise RuntimeError("data validation failed")
        return {"validated": True}

    def post_data_validation(self, report):
        """Post-data validation hook."""
        self.data_validation_report = report

    def on_data_validation_error(self, error: Exception) -> None:
        """Error hook."""
        self.data_validation_failure_called = True

    def post_model_governance(self, result: TrainingResult, report: dict) -> None:
        """Post-model governance hook."""
        self.model_governance_called = True


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
    tracker = RecordingTracker()
    trainer = DummyTrainer(cfg, tracker=tracker)
    assert trainer.get_training_table_name() == "DB.SC.T"

    result = trainer.execute_training(data="dummy")

    assert result.training_status == TrainingStatus.SUCCESS
    assert "duration_seconds" in result.metrics
    assert tracker.events[-1][1] == "training_end"
    assert trainer.data_validation_pre_called is True
    assert trainer.data_validation_report == {"validated": True}
    assert trainer.model_governance_called is True
    assert trainer.data_validation_failure_called is False


def test_training_data_validation_failure():
    """Ensure training data governance failure hook executes on error."""
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
    trainer.raise_data_validation_error = True

    with pytest.raises(RuntimeError, match="data validation failed"):
        trainer.execute_training(data="dummy")

    assert trainer.data_validation_pre_called is True
    assert trainer.data_validation_failure_called is True
