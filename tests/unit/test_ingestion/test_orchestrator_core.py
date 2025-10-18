"""Unit tests for ingestion orchestrator."""

import pytest

from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    DataSource,
    ExecutionEventTracker,
    IngestionConfig,
    IngestionMethod,
    IngestionResult,
    SourceType,
)
from snowflake_ml_template.ingestion.orchestrator import IngestionOrchestrator


class DummyStrategy(BaseIngestionStrategy):
    """Dummy strategy for testing."""

    def ingest(self, source, target, **kwargs):
        """Ingest data."""
        return IngestionResult(
            status="success",
            method=self.config.method,
            target_table=target,
            rows_loaded=10,
        )

    def validate(self) -> bool:
        """Validate."""
        return True


class FailingStrategy(BaseIngestionStrategy):
    """Failing strategy for testing."""

    def ingest(self, source, target, **kwargs):  # pragma: no cover
        """Ingest data."""
        raise RuntimeError("validation should prevent ingestion")

    def validate(self) -> bool:
        """Validate."""
        return False


class StubSession:
    """Stub session."""

    pass


def make_config():
    """Make config."""
    ds = DataSource(source_type=SourceType.S3, location="s3://b", file_format="CSV")
    return IngestionConfig(
        method=IngestionMethod.SNOWPIPE,
        source=ds,
        target_database="DB",
        target_schema="SC",
        target_table="T",
        warehouse="WH",
    )


class TrackerStub(ExecutionEventTracker):
    """Capture events for verification."""

    def __init__(self) -> None:
        """Initialize."""
        self.events: list[tuple[str, str]] = []

    def record_event(self, component: str, event: str, payload: dict) -> None:
        """Record event."""
        self.events.append((component, event))


def test_register_and_execute_success():
    """Test register and execute success."""
    tracker = TrackerStub()
    orch = IngestionOrchestrator(StubSession(), tracker=tracker)
    strat = DummyStrategy(make_config())
    orch.register_strategy("snowpipe", strat)
    res = orch.execute("snowpipe")
    assert res.status == "success" and res.rows_loaded == 10
    assert ("DummyStrategy", "ingestion_start") in tracker.events


def test_execute_unknown_strategy_raises():
    """Test execute unknown strategy raises."""
    orch = IngestionOrchestrator(StubSession())
    with pytest.raises(ValueError):
        orch.execute("missing")


def test_execute_validation_failed_raises():
    """Test execute validation failed raises."""
    orch = IngestionOrchestrator(StubSession())
    strat = FailingStrategy(make_config())
    orch.register_strategy("fail", strat)
    # Override validate to fail
    strat.validate = lambda: False
    with pytest.raises(ValueError):
        orch.execute("fail")
