"""Tests for base transformation."""

from datetime import datetime, timedelta

import pytest

from snowflake_ml_template.core.base.transformation import (
    BaseTransformation,
    TransformationConfig,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)


class RecordingTracker:
    """Simple tracker capture for tests."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.events = []

    def record_event(self, component: str, event: str, payload: dict) -> None:
        """Record an event."""
        self.events.append((component, event, payload))


def test_transformation_config_validation():
    """Test transformation config validation."""
    with pytest.raises(ValueError):
        TransformationConfig(
            transformation_type=TransformationType.SQL,
            source_database="",
            source_schema="SC",
            source_table="S",
            target_database="DB",
            target_schema="SC",
            target_table="T",
            warehouse="WH",
        )
    with pytest.raises(ValueError):
        TransformationConfig(
            transformation_type=TransformationType.SQL,
            source_database="DB",
            source_schema="SC",
            source_table="S",
            target_database="",
            target_schema="SC",
            target_table="T",
            warehouse="WH",
        )
    with pytest.raises(ValueError):
        TransformationConfig(
            transformation_type=TransformationType.SQL,
            source_database="DB",
            source_schema="SC",
            source_table="S",
            target_database="DB",
            target_schema="SC",
            target_table="T",
            warehouse="",
        )
    with pytest.raises(ValueError):
        TransformationConfig(
            transformation_type=TransformationType.SQL,
            source_database="DB",
            source_schema="SC",
            source_table="S",
            target_database="DB",
            target_schema="SC",
            target_table="T",
            warehouse="WH",
            mode="invalid",
        )

    ok = TransformationConfig(
        transformation_type=TransformationType.SQL,
        source_database="SDB",
        source_schema="SSC",
        source_table="ST",
        target_database="TDB",
        target_schema="TSC",
        target_table="TT",
        warehouse="WH",
        mode="append",
    )
    assert ok.mode == "append"


def test_transformation_result_duration():
    """Test transformation result duration."""
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(seconds=7)
    res = TransformationResult(
        status="success",
        transformation_type=TransformationType.SQL,
        target_table="DB.SC.T",
        start_time=start,
        end_time=end,
    )
    assert res.duration_seconds == 7.0


class DummyTransformation(BaseTransformation):
    """Dummy transformation for testing."""

    def __init__(self, config: TransformationConfig, tracker=None) -> None:
        """Initialize the transformation."""
        super().__init__(config, tracker=tracker)
        self.pre_called = False
        self.post_called = False
        self.error_called = False

    def transform(self, **kwargs):  # pragma: no cover
        """Return successful transformation result."""
        return TransformationResult(
            status=TransformationStatus.SUCCESS,
            transformation_type=self.config.transformation_type,
            target_table=self.get_target_table_name(),
            rows_processed=100,
            rows_written=95,
        )

    def validate(self) -> bool:  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def pre_transform(self) -> None:
        """Set pre_called to True."""
        self.pre_called = True

    def post_transform(self, result: TransformationResult) -> None:
        """Set post_called to True."""
        self.post_called = True

    def on_transform_error(self, error: Exception) -> None:  # pragma: no cover
        """Set error_called to True."""
        self.error_called = True


def test_base_transformation_helpers():
    """Test base transformation helpers."""
    cfg = TransformationConfig(
        transformation_type=TransformationType.SQL,
        source_database="SDB",
        source_schema="SSC",
        source_table="ST",
        target_database="TDB",
        target_schema="TSC",
        target_table="TT",
        warehouse="WH",
    )
    tracker = RecordingTracker()
    t = DummyTransformation(cfg, tracker=tracker)
    assert t.get_source_table_name() == "SDB.SSC.ST"
    assert t.get_target_table_name() == "TDB.TSC.TT"

    result = t.execute_transformation()

    assert result.transformation_status == TransformationStatus.SUCCESS
    assert "duration_seconds" in result.metrics
    assert t.pre_called is True
    assert t.post_called is True
    assert tracker.events[-1][1] == "transformation_end"
