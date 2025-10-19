"""Tests for feature quality monitoring."""

from unittest.mock import Mock

import pytest

from snowflake_ml_template.feature_store.monitoring.quality import (
    FeatureQualityMonitor,
    QualityMetrics,
)


def test_quality_monitor_initialization(mock_session):
    """Test quality monitor initialization."""
    monitor = FeatureQualityMonitor(mock_session)
    assert monitor.session == mock_session


def test_quality_monitor_check_nulls(mock_session):
    """Test null checking."""
    monitor = FeatureQualityMonitor(mock_session)

    # Mock dataframe
    df = Mock()
    filtered_df = Mock()
    df.filter.return_value = filtered_df
    filtered_df.count.return_value = 5

    result = monitor.check_nulls(df, "feature_col")

    assert result == 5
    df.filter.assert_called_once()
    filtered_df.count.assert_called_once()


def test_record_metrics_writes_insert(mock_session):
    """Verify record_metrics issues an INSERT when persistence configured."""
    session = Mock()
    session.sql.return_value.collect.return_value = []
    monitor = FeatureQualityMonitor(
        session,
        database="DB",
        schema="MONITOR",
        table="QUALITY_EVENTS",
    )

    metrics = QualityMetrics(
        feature_name="F1",
        total_rows=100,
        null_count=5,
        null_rate=0.05,
        unique_count=50,
        mean=1.0,
        std=0.5,
        min=0.0,
        max=2.0,
        quality_score=0.9,
    )

    session.sql.reset_mock()
    monitor.record_metrics(metrics, feature_view="fv", entity="customer", run_id="abc")

    executed_sql = session.sql.call_args[0][0]
    assert "INSERT INTO DB.MONITOR.QUALITY_EVENTS" in executed_sql
    assert "'fv'" in executed_sql
    assert "'F1'" in executed_sql


def test_create_quality_task_requires_events_table(mock_session):
    """Ensure task creation fails without persistence configured."""
    monitor = FeatureQualityMonitor(mock_session)
    with pytest.raises(Exception):
        monitor.create_quality_task(
            task_name="TASK",
            warehouse="WH",
            schedule="1 minute",
            procedure_call="CALL RUN()",
        )


def test_create_quality_task_emits_sql(mock_session):
    """Verify task creation emits proper SQL when configured."""
    session = Mock()
    session.sql.return_value.collect.return_value = []
    monitor = FeatureQualityMonitor(
        session,
        database="DB",
        schema="MONITOR",
        table="QUALITY_EVENTS",
    )

    session.sql.reset_mock()
    monitor.create_quality_task(
        task_name="DB.MONITOR.QUALITY_TASK",
        warehouse="MONITOR_WH",
        schedule="USING CRON * * * * * UTC",
        procedure_call="CALL DB.MONITOR.RUN_QUALITY_CHECK()",
    )

    executed_sql = session.sql.call_args[0][0]
    assert "CREATE OR REPLACE TASK DB.MONITOR.QUALITY_TASK" in executed_sql
