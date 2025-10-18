"""Tests for base transformation."""

from datetime import datetime, timedelta

import pytest

from snowflake_ml_template.core.base.transformation import (
    BaseTransformation,
    TransformationConfig,
    TransformationResult,
    TransformationType,
)


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

    def transform(self, **kwargs):  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def validate(self) -> bool:  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError


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
    t = DummyTransformation(cfg)
    assert t.get_source_table_name() == "SDB.SSC.ST"
    assert t.get_target_table_name() == "TDB.TSC.TT"
