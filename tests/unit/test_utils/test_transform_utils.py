"""
Unit tests for transform_utils.py.

These tests validate the functionality of the Snowpark transformation utilities.
"""

from unittest.mock import MagicMock

import pytest
from snowflake.snowpark import Row

from src.snowflake_ml_template.utils.transform_utils import (
    add_row_number,
    deduplicate_dataframe,
    fill_nulls,
    filter_by_condition,
    log_dataframe_info,
    pivot_dataframe,
    rename_columns,
    sample_dataframe,
    unpivot_dataframe,
    validate_schema,
)


@pytest.fixture(scope="module")
def snowpark_session():
    """Fixture to create a mocked Snowpark session for testing."""
    session = MagicMock()
    session.create_dataframe = MagicMock()
    return session


@pytest.mark.parametrize(
    "input_schema,expected_schema,expected_result",
    [
        (
            {"name": "STRING", "age": "INTEGER"},
            {"name": "STRING", "age": "INTEGER"},
            True,
        ),
        ({"name": "STRING"}, {"name": "STRING", "age": "INTEGER"}, False),
        (
            {"name": "STRING", "age": "FLOAT"},
            {"name": "STRING", "age": "INTEGER"},
            False,
        ),
        ({}, {"name": "STRING", "age": "INTEGER"}, False),
    ],
)
def test_validate_schema_param(
    input_schema, expected_schema, expected_result, snowpark_session
):
    """Test validate_schema with various input and expected schemas."""
    df = snowpark_session.create_dataframe.return_value

    # Enhance mock setup to ensure datatype values are correctly mocked
    mocked_fields = []
    for col, type_ in input_schema.items():
        field = MagicMock()
        field.name = col
        field.datatype = type_
        mocked_fields.append(field)

    df.schema.fields = mocked_fields
    assert validate_schema(df, expected_schema) == expected_result


def test_validate_schema(snowpark_session):
    """Test validate_schema with a matching schema."""
    df = snowpark_session.create_dataframe.return_value

    # Fully mock the StructField objects with correct datatype behavior
    mocked_fields = [
        MagicMock(name="name", datatype="STRING"),
        MagicMock(name="age", datatype="INTEGER"),
    ]
    for field in mocked_fields:
        field.name = field._mock_name
    df.schema.fields = mocked_fields

    expected_schema = {"name": "STRING", "age": "INTEGER"}
    assert validate_schema(df, expected_schema) is True


def test_pivot_dataframe(snowpark_session):
    """Test pivot_dataframe with valid parameters."""
    df = snowpark_session.create_dataframe.return_value
    df.pivot.return_value.agg.return_value = df
    pivoted_df = pivot_dataframe(df, "month", "sales", ["name"])
    assert pivoted_df == df


def test_pivot_dataframe_invalid_column(snowpark_session):
    """Test pivot_dataframe with an invalid pivot_column."""
    df = snowpark_session.create_dataframe.return_value
    df.pivot.side_effect = KeyError("Invalid column")
    with pytest.raises(KeyError, match="Invalid column"):
        pivot_dataframe(df, "invalid_column", "sales", ["name"])


def test_unpivot_dataframe(snowpark_session):
    """Test unpivot_dataframe with valid parameters."""
    df = snowpark_session.create_dataframe.return_value
    df.unpivot.return_value = df
    unpivoted_df = unpivot_dataframe(df, ["2025-01", "2025-02"], "month", "sales")
    assert unpivoted_df == df


def test_unpivot_dataframe_empty_columns(snowpark_session):
    """Test unpivot_dataframe with an empty list of unpivot_columns."""
    df = snowpark_session.create_dataframe.return_value
    with pytest.raises(ValueError, match="unpivot_columns cannot be empty"):
        unpivot_dataframe(df, [], "month", "sales")


def test_deduplicate_dataframe(snowpark_session):
    """Test deduplicate_dataframe with a valid subset."""
    df = snowpark_session.create_dataframe.return_value
    df.drop_duplicates.return_value = df
    deduplicated_df = deduplicate_dataframe(df, ["name", "age"])
    assert deduplicated_df == df


def test_deduplicate_dataframe_empty_subset(snowpark_session):
    """Test deduplicate_dataframe with an empty subset."""
    df = snowpark_session.create_dataframe.return_value
    with pytest.raises(ValueError, match="subset cannot be empty"):
        deduplicate_dataframe(df, [])


def test_log_dataframe_info(snowpark_session, capsys):
    """Test log_dataframe_info to ensure it runs without error."""
    df = snowpark_session.create_dataframe.return_value
    df.schema = "Mocked Schema"
    df.limit.return_value.collect.return_value = [Row("Alice", 30)]
    log_dataframe_info(df, num_rows=1)
    captured = capsys.readouterr()
    assert "Schema:" in captured.out
    assert "Sample Data:" in captured.out


def test_log_dataframe_info_content(snowpark_session, capsys):
    """Test log_dataframe_info to validate logged content."""
    df = snowpark_session.create_dataframe.return_value
    df.schema = "Mocked Schema"
    df.limit.return_value.collect.return_value = [Row("Alice", 30)]
    log_dataframe_info(df, num_rows=1)
    captured = capsys.readouterr()
    assert "Schema:" in captured.out
    assert "Mocked Schema" in captured.out
    assert "Sample Data:" in captured.out
    assert "[Row('Alice', 30)]" in captured.out


def test_fill_nulls(snowpark_session):
    """Test fill_nulls function."""
    df = snowpark_session.create_dataframe.return_value
    df.na.fill.return_value = df
    filled_df = fill_nulls(df, {"age": 0, "name": "Unknown"})
    assert filled_df == df


def test_rename_columns(snowpark_session):
    """Test rename_columns function."""
    df = snowpark_session.create_dataframe.return_value
    df.withColumnRenamed.return_value = df
    renamed_df = rename_columns(df, {"old_name": "new_name"})
    assert renamed_df == df


def test_sample_dataframe(snowpark_session):
    """Test sample_dataframe function."""
    df = snowpark_session.create_dataframe.return_value
    df.sample.return_value = df
    sampled_df = sample_dataframe(df, 0.5, seed=42)
    assert sampled_df == df


def test_sample_dataframe_invalid_fraction(snowpark_session):
    """Test sample_dataframe with invalid fraction."""
    df = snowpark_session.create_dataframe.return_value
    with pytest.raises(ValueError, match="fraction must be between 0 and 1"):
        sample_dataframe(df, 1.5)


def test_add_row_number(snowpark_session):
    """Test add_row_number function."""
    df = snowpark_session.create_dataframe.return_value
    df.withColumn.return_value = df
    numbered_df = add_row_number(df, partition_by=["group"], order_by=["id"])
    assert numbered_df == df


def test_filter_by_condition(snowpark_session):
    """Test filter_by_condition function."""
    df = snowpark_session.create_dataframe.return_value
    df.filter.return_value = df
    filtered_df = filter_by_condition(df, "age > 18")
    assert filtered_df == df
