"""
Unit tests for data_quality_utils.py.

These tests validate the functionality of data quality utilities.
"""

from unittest.mock import MagicMock

import pytest
from snowflake.snowpark import Row
from snowflake.snowpark.types import IntegerType, StringType, StructField

from src.snowflake_ml_template.utils.data_quality_utils import (
    check_data_completeness,
    detect_outliers,
    profile_dataframe,
    validate_data_types,
)


@pytest.fixture(scope="module")
def snowpark_session():
    """Fixture to create a mocked Snowpark session for testing."""
    session = MagicMock()
    session.create_dataframe = MagicMock()
    return session


def test_profile_dataframe(snowpark_session):
    """Test profile_dataframe function."""
    df = snowpark_session.create_dataframe.return_value
    df.schema.fields = [
        StructField("name", StringType()),
        StructField("age", IntegerType()),
    ]
    df.select.return_value.collect.return_value = [
        Row(
            TOTAL_COUNT=100,
            DISTINCT_COUNT=50,
            NULL_COUNT=5,
            MEAN=30.0,
            STDDEV=5.0,
            MIN=20,
            MAX=40,
        )
    ]
    profile = profile_dataframe(df)
    assert "NAME" in profile
    assert "AGE" in profile


def test_detect_outliers(snowpark_session):
    """Test detect_outliers function."""
    df = snowpark_session.create_dataframe.return_value
    df.select.return_value.collect.return_value = [
        MagicMock(__getitem__=lambda self, key: 25.0 if key == "Q1" else 75.0)
    ]
    df.withColumn.return_value = df
    outlier_df = detect_outliers(df, "age")
    assert outlier_df == df


def test_check_data_completeness(snowpark_session):
    """Test check_data_completeness function."""
    df = snowpark_session.create_dataframe.return_value
    df.count.return_value = 100
    df.filter.return_value.count.return_value = 95
    is_complete, ratios = check_data_completeness(df, threshold=0.9)
    assert is_complete is True
    assert isinstance(ratios, dict)


def test_validate_data_types(snowpark_session):
    """Test validate_data_types function."""
    df = snowpark_session.create_dataframe.return_value

    # Mock fields with correct datatype
    mock_field1 = MagicMock()
    mock_field1.name = "name"
    mock_field1.datatype.__class__.__name__ = "StringType"

    mock_field2 = MagicMock()
    mock_field2.name = "age"
    mock_field2.datatype.__class__.__name__ = "IntegerType"

    df.schema.fields = [mock_field1, mock_field2]

    is_valid, errors = validate_data_types(df, {"name": "STRING", "age": "INTEGER"})
    assert is_valid is True
    assert errors == []
