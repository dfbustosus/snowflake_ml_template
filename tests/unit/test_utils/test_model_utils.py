"""Tests for model utilities."""

from unittest.mock import MagicMock

import pytest

from snowflake_ml_template.utils.model_utils import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    validate_model_predictions,
)


@pytest.fixture(scope="module")
def snowpark_session():
    """Fixture to create a mocked Snowpark session for testing."""
    session = MagicMock()
    session.create_dataframe = MagicMock()
    return session


class TestModelUtils:
    """Test cases for model utilities."""

    def test_calculate_classification_metrics(self, snowpark_session):
        """Test calculate_classification_metrics function."""
        df = snowpark_session.create_dataframe.return_value
        df.count.return_value = 100
        df.filter.return_value.count.side_effect = [
            80,
            20,
            10,
            10,
            5,
        ]  # tp, tn, fp, fn, correct

        metrics = calculate_classification_metrics(df, "actual", "predicted")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_calculate_regression_metrics(self, snowpark_session):
        """Test calculate_regression_metrics function."""
        df = snowpark_session.create_dataframe.return_value
        df.select.return_value.agg.return_value.collect.return_value = [
            {
                "COUNT(*)": 100,
                "AVG(SQUARED_ERROR)": 4.0,
                "AVG(ABS_ERROR)": 2.0,
                "AVG(ACTUAL)": 10.0,
                "AVG(PREDICTED)": 10.0,
            }
        ]
        df.select.return_value.collect.return_value = [{"ACTUAL": 10.0}]

        metrics = calculate_regression_metrics(df, "actual", "predicted")
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r_squared" in metrics

    def test_validate_model_predictions(self, snowpark_session):
        """Test validate_model_predictions function."""
        df = snowpark_session.create_dataframe.return_value
        df.count.return_value = 100
        df.filter.return_value.count.return_value = 0  # No nulls

        # Mock classification metrics
        df.filter.side_effect = lambda cond: MagicMock(
            count=MagicMock(return_value=80)
        )  # correct predictions

        result = validate_model_predictions(df, "actual", "predicted", "classification")
        assert "total_predictions" in result
        assert "metrics" in result
        assert "data_quality_issues" in result
