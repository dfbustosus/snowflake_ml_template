"""Tests for monitoring components."""

from snowflake_ml_template.monitoring.data_monitor import DataMonitor
from snowflake_ml_template.monitoring.infrastructure_monitor import (
    InfrastructureMonitor,
)
from snowflake_ml_template.monitoring.model_monitor import ModelMetrics, ModelMonitor


def test_model_monitor_initialization(mock_session):
    """Test model monitor initialization."""
    monitor = ModelMonitor(mock_session, "TEST_DB", "MONITORING")
    assert monitor.session == mock_session
    assert monitor.database == "TEST_DB"


def test_model_monitor_log_metrics(mock_session):
    """Test model monitor logs metrics."""
    monitor = ModelMonitor(mock_session, "TEST_DB", "MONITORING")
    metrics = ModelMetrics(
        model_name="test_model",
        version="1.0.0",
        accuracy=0.95,
        precision=0.93,
        recall=0.92,
        f1_score=0.925,
        auc=0.96,
    )
    monitor.log_metrics(metrics)
    assert True


def test_data_monitor_initialization(mock_session):
    """Test data monitor initialization."""
    monitor = DataMonitor(mock_session, "TEST_DB", "MONITORING")
    assert monitor.session == mock_session


def test_data_monitor_check_quality(mock_session):
    """Test data monitor checks quality."""
    mock_session.table.return_value.count.return_value = 1000
    monitor = DataMonitor(mock_session, "TEST_DB", "MONITORING")
    result = monitor.check_data_quality("TEST_TABLE")
    assert result["total_rows"] == 1000


def test_infrastructure_monitor_initialization(mock_session):
    """Test infrastructure monitor initialization."""
    monitor = InfrastructureMonitor(mock_session)
    assert monitor.session == mock_session


def test_infrastructure_monitor_warehouse_usage(mock_session):
    """Test infrastructure monitor gets warehouse usage."""
    mock_session.sql.return_value.collect.return_value = [[100.5]]
    monitor = InfrastructureMonitor(mock_session)
    result = monitor.get_warehouse_usage("TEST_WH")
    assert result["warehouse"] == "TEST_WH"
    assert result["credits_7d"] == 100.5
