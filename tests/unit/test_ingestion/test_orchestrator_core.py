"""Unit tests for ingestion orchestrator."""

import pytest

from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    DataSource,
    IngestionConfig,
    IngestionMethod,
    IngestionResult,
    SourceType,
)
from snowflake_ml_template.ingestion.orchestrator import IngestionOrchestrator


class DummyStrategy(BaseIngestionStrategy):
    """Dummy strategy for testing."""

    def set_session(self, session):
        """Set session."""
        self._session = session

    def ingest(self, source, target, **kwargs):
        """Ingest."""
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

    def set_session(self, session):
        """Set session."""
        self._session = session

    def ingest(self, source, target, **kwargs):  # pragma: no cover
        """Ingest."""
        return IngestionResult(
            status="failed", method=self.config.method, target_table=target
        )

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


def test_register_and_execute_success():
    """Test register and execute success."""
    orch = IngestionOrchestrator(StubSession())
    strat = DummyStrategy(make_config())
    orch.register_strategy("snowpipe", strat)
    res = orch.execute("snowpipe")
    assert res.status == "success" and res.rows_loaded == 10


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
    with pytest.raises(ValueError):
        orch.execute("fail")
