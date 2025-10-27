"""Tests for feature selection utilities."""

from unittest.mock import MagicMock

import pytest

from snowflake_ml_template.utils.feature_selection_utils import (
    remove_multicollinear_features,
    select_features_by_correlation,
    select_features_by_missing_values,
    select_features_by_variance,
)


@pytest.fixture(scope="module")
def snowpark_session():
    """Fixture to create a mocked Snowpark session for testing."""
    session = MagicMock()
    session.create_dataframe = MagicMock()
    return session


class TestFeatureSelectionUtils:
    """Test cases for feature selection utilities."""

    def test_select_features_by_correlation(self, snowpark_session):
        """Test select_features_by_correlation function."""
        df = snowpark_session.create_dataframe.return_value
        df.select.return_value.collect.return_value = [{"CORRELATION": 0.9}]

        # Mock schema
        mock_field1 = MagicMock()
        mock_field1.name = "feature1"
        mock_field1.datatype.__class__.__name__ = "FloatType"

        mock_field2 = MagicMock()
        mock_field2.name = "target"
        mock_field2.datatype.__class__.__name__ = "IntegerType"

        df.schema.fields = [mock_field1, mock_field2]

        features = select_features_by_correlation(df, "target", threshold=0.8)
        assert "feature1" in features

    def test_select_features_by_variance(self, snowpark_session):
        """Test select_features_by_variance function."""
        df = snowpark_session.create_dataframe.return_value
        df.select.return_value.collect.return_value = [{"VARIANCE": 0.1}]

        # Mock schema
        mock_field = MagicMock()
        mock_field.name = "feature1"
        mock_field.datatype.__class__.__name__ = "FloatType"

        df.schema.fields = [mock_field]

        features = select_features_by_variance(df, threshold=0.05)
        assert "feature1" in features

    def test_remove_multicollinear_features(self, snowpark_session):
        """Test remove_multicollinear_features function."""
        df = snowpark_session.create_dataframe.return_value
        df.select.return_value.collect.return_value = [{"CORR": 0.95}]

        # Mock schema
        mock_field1 = MagicMock()
        mock_field1.name = "feature1"
        mock_field1.datatype.__class__.__name__ = "FloatType"

        mock_field2 = MagicMock()
        mock_field2.name = "feature2"
        mock_field2.datatype.__class__.__name__ = "FloatType"

        df.schema.fields = [mock_field1, mock_field2]

        features = remove_multicollinear_features(df, threshold=0.9)
        assert len(features) == 1  # One should be removed

    def test_select_features_by_missing_values(self, snowpark_session):
        """Test select_features_by_missing_values function."""
        df = snowpark_session.create_dataframe.return_value
        df.count.return_value = 100
        df.filter.return_value.count.return_value = 95

        # Mock schema
        mock_field = MagicMock()
        mock_field.name = "feature1"
        mock_field.datatype.__class__.__name__ = "StringType"

        df.schema.fields = [mock_field]

        features = select_features_by_missing_values(df, threshold=0.1)
        assert "feature1" in features
