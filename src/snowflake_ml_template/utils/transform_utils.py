"""Utility functions for Snowpark transformations.

This module contains reusable functions for common data transformations and validations
using Snowpark DataFrames.
"""

from typing import Dict, List

from snowflake.snowpark import DataFrame


def validate_schema(df: DataFrame, expected_schema: Dict[str, str]) -> bool:
    """Validate that a Snowpark DataFrame matches the expected schema.

    Args:
        df (DataFrame): The Snowpark DataFrame to validate.
        expected_schema (Dict[str, str]): A dictionary where keys are column names
            and values are expected data types (e.g., 'STRING', 'NUMBER').

    Returns:
        bool: True if the schema matches, False otherwise.
    """
    actual_schema = {
        field.name: str(field.datatype).upper() for field in df.schema.fields
    }

    for column, expected_type in expected_schema.items():
        if (
            column not in actual_schema
            or actual_schema[column] != expected_type.upper()
        ):
            return False
    return True


def pivot_dataframe(
    df: DataFrame, pivot_column: str, value_column: str, group_by_columns: List[str]
) -> DataFrame:
    """Perform a pivot operation on a Snowpark DataFrame.

    Args:
        df (DataFrame): The Snowpark DataFrame to pivot.
        pivot_column (str): The column to pivot.
        value_column (str): The column containing values to aggregate.
        group_by_columns (List[str]): Columns to group by.

    Returns:
        DataFrame: The pivoted DataFrame.
    """
    return df.pivot(pivot_column, group_by_columns).agg({value_column: "sum"})


def unpivot_dataframe(
    df: DataFrame, unpivot_columns: List[str], key_column: str, value_column: str
) -> DataFrame:
    """Perform an unpivot operation on a Snowpark DataFrame.

    Args:
        df (DataFrame): The Snowpark DataFrame to unpivot.
        unpivot_columns (List[str]): Columns to unpivot.
        key_column (str): The name of the new key column.
        value_column (str): The name of the new value column.

    Returns:
        DataFrame: The unpivoted DataFrame.

    Raises:
        ValueError: If unpivot_columns is empty.
    """
    if not unpivot_columns:
        raise ValueError("unpivot_columns cannot be empty")
    return df.unpivot(unpivot_columns, key_column, value_column)


def deduplicate_dataframe(df: DataFrame, subset: List[str]) -> DataFrame:
    """Remove duplicate rows from a Snowpark DataFrame based on a subset of columns.

    Args:
        df (DataFrame): The Snowpark DataFrame to deduplicate.
        subset (List[str]): Columns to consider for identifying duplicates.

    Returns:
        DataFrame: The deduplicated DataFrame.

    Raises:
        ValueError: If subset is empty.
    """
    if not subset:
        raise ValueError("subset cannot be empty")
    return df.drop_duplicates(subset)


def log_dataframe_info(df: DataFrame, num_rows: int = 5) -> None:
    """Log the schema and a sample of rows from a Snowpark DataFrame.

    Args:
        df (DataFrame): The Snowpark DataFrame to log.
        num_rows (int): Number of rows to display.
    """
    print("Schema:")
    print(df.schema)
    print("\nSample Data:")
    print(df.limit(num_rows).collect())
