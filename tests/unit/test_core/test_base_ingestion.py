"""Tests for base ingestion."""

from datetime import datetime, timedelta

import pytest

from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    DataSource,
    IngestionConfig,
    IngestionMethod,
    IngestionResult,
    SourceType,
)


def test_datasource_validation():
    """Test datasource validation."""
    with pytest.raises(ValueError):
        DataSource(source_type=SourceType.S3, location="", file_format="CSV")
    with pytest.raises(ValueError):
        DataSource(source_type=SourceType.S3, location="s3://bucket", file_format="")

    ds = DataSource(source_type=SourceType.S3, location="s3://b", file_format="CSV")
    assert ds.location == "s3://b"


def test_ingestion_config_validation():
    """Test ingestion config validation."""
    ds = DataSource(source_type=SourceType.S3, location="s3://b", file_format="CSV")
    with pytest.raises(ValueError):
        IngestionConfig(
            method=IngestionMethod.SNOWPIPE,
            source=ds,
            target_database="",
            target_schema="SC",
            target_table="T",
            warehouse="WH",
        )
    with pytest.raises(ValueError):
        IngestionConfig(
            method=IngestionMethod.SNOWPIPE,
            source=ds,
            target_database="DB",
            target_schema="",
            target_table="T",
            warehouse="WH",
        )
    with pytest.raises(ValueError):
        IngestionConfig(
            method=IngestionMethod.SNOWPIPE,
            source=ds,
            target_database="DB",
            target_schema="SC",
            target_table="",
            warehouse="WH",
        )
    with pytest.raises(ValueError):
        IngestionConfig(
            method=IngestionMethod.SNOWPIPE,
            source=ds,
            target_database="DB",
            target_schema="SC",
            target_table="T",
            warehouse="",
        )
    with pytest.raises(ValueError):
        IngestionConfig(
            method=IngestionMethod.SNOWPIPE,
            source=ds,
            target_database="DB",
            target_schema="SC",
            target_table="T",
            warehouse="WH",
            on_error="BAD",
        )

    ok = IngestionConfig(
        method=IngestionMethod.SNOWPIPE,
        source=ds,
        target_database="DB",
        target_schema="SC",
        target_table="T",
        warehouse="WH",
        on_error="ABORT_STATEMENT",
    )
    assert ok.on_error == "ABORT_STATEMENT"


def test_ingestion_result_duration():
    """Test ingestion result duration."""
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(seconds=5)
    res = IngestionResult(
        status="success",
        method=IngestionMethod.SNOWPIPE,
        target_table="DB.SC.T",
        start_time=start,
        end_time=end,
    )
    assert res.duration_seconds == 5.0


class DummyStrategy(BaseIngestionStrategy):
    """Dummy strategy for testing."""

    def ingest(self, source, target, **kwargs):  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def validate(self) -> bool:  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError


def test_base_ingestion_strategy_helpers():
    """Test base ingestion strategy helpers."""
    ds = DataSource(source_type=SourceType.S3, location="s3://b", file_format="CSV")
    cfg = IngestionConfig(
        method=IngestionMethod.SNOWPIPE,
        source=ds,
        target_database="DB",
        target_schema="SC",
        target_table="T",
        warehouse="WH",
    )
    strat = DummyStrategy(cfg)
    assert strat.get_target_table_name() == "DB.SC.T"
