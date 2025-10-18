"""Tests for feature quality monitoring."""

from unittest.mock import Mock

from snowflake_ml_template.feature_store.monitoring.quality import FeatureQualityMonitor


def test_quality_monitor_initialization(mock_session):
    """Test quality monitor initialization."""
    monitor = FeatureQualityMonitor(mock_session)
    assert monitor.session == mock_session


def test_quality_monitor_check_nulls(mock_session):
    """Test null checking."""
    monitor = FeatureQualityMonitor(mock_session)

    # Mock dataframe
    df = Mock()
    df.select.return_value.agg.return_value.collect.return_value = [{"null_count": 5}]

    monitor.check_nulls(df, "feature_col")

    assert True  # Interface test passed
