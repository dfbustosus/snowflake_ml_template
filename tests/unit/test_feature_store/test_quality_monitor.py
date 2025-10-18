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
    filtered_df = Mock()
    df.filter.return_value = filtered_df
    filtered_df.count.return_value = 5

    result = monitor.check_nulls(df, "feature_col")

    assert result == 5
    df.filter.assert_called_once()
    filtered_df.count.assert_called_once()
