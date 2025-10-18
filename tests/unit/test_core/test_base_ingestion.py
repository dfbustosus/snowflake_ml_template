"""Tests for base ingestion."""

from datetime import datetime, timedelta

import pytest

from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    DataSource,
    IngestionConfig,
    IngestionMethod,
    IngestionResult,
    IngestionStatus,
    SourceType,
)


class RecordingTracker:
    """Simple tracker implementation for tests."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.events = []

    def record_event(self, component: str, event: str, payload: dict) -> None:
        """Record an event."""
        self.events.append((component, event, payload))


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

    def __init__(self, config: IngestionConfig, tracker=None) -> None:
        """Initialize the dummy strategy."""
        super().__init__(config, tracker=tracker)
        self.pre_called = False
        self.post_called = False
        self.error_called = False
        self.validation_pre_called = False
        self.validation_post_report: dict | None = None
        self.validation_failure_called = False
        self.raise_validation_error = False

    def ingest(self, source, target, **kwargs):  # pragma: no cover
        """Return a successful ingestion result for testing."""
        return IngestionResult(
            status=IngestionStatus.SUCCESS,
            method=self.config.method,
            target_table=target,
            rows_loaded=10,
            files_processed=1,
        )

    def validate(self) -> bool:  # pragma: no cover
        """Raise NotImplementedError."""
        raise NotImplementedError

    def pre_ingest(self, source):
        """Pre-ingest hook."""
        self.pre_called = True

    def post_ingest(self, result: IngestionResult):
        """Post-ingest hook."""
        self.post_called = True

    def on_ingest_error(self, error: Exception):  # pragma: no cover
        """Error hook."""
        self.error_called = True

    def pre_validation(self, source: DataSource) -> None:
        """Pre-validation hook."""
        self.validation_pre_called = True

    def validate_source(self, source: DataSource) -> dict:
        """Validate source hook."""
        if self.raise_validation_error:
            raise RuntimeError("validation failed")
        return {"validated": True}

    def post_validation(self, source: DataSource, report: dict) -> None:
        """Post-validation hook."""
        self.validation_post_report = report

    def on_validation_error(self, source: DataSource, error: Exception) -> None:
        """Error hook."""
        self.validation_failure_called = True


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
    tracker = RecordingTracker()
    strat = DummyStrategy(cfg, tracker=tracker)
    assert strat.get_target_table_name() == "DB.SC.T"

    result = strat.execute_ingestion(ds, strat.get_target_table_name())

    assert result.ingestion_status == IngestionStatus.SUCCESS
    assert result.metrics["duration_seconds"] >= 0
    assert strat.pre_called is True
    assert strat.post_called is True
    assert tracker.events[-1][1] == "ingestion_end"
    assert strat.validation_pre_called is True
    assert strat.validation_post_report == {"validated": True}
    assert strat.validation_failure_called is False


def test_ingestion_validation_failure(monkeypatch):
    """Ensure validation failure hook executes and exception propagates."""
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
    strat.raise_validation_error = True

    with pytest.raises(RuntimeError, match="validation failed"):
        strat.execute_ingestion(ds, strat.get_target_table_name())

    assert strat.validation_pre_called is True
    assert strat.validation_failure_called is True
