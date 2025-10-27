"""Utility functions for Snowpark transformations.

This module contains reusable functions for common data transformations and validations
using Snowpark DataFrames.
"""

from typing import Any, Dict, List, Optional

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


def fill_nulls(df: DataFrame, fill_values: Dict[str, Any]) -> DataFrame:
    """Fill null values in specified columns with given values.

    Args:
        df (DataFrame): The Snowpark DataFrame to fill nulls in.
        fill_values (Dict[str, Any]): A dictionary where keys are column names
            and values are the fill values.

    Returns:
        DataFrame: The DataFrame with nulls filled.
    """
    for column, value in fill_values.items():
        df = df.na.fill({column: value})
    return df


def rename_columns(df: DataFrame, column_mapping: Dict[str, str]) -> DataFrame:
    """Rename multiple columns in a Snowpark DataFrame.

    Args:
        df (DataFrame): The Snowpark DataFrame to rename columns in.
        column_mapping (Dict[str, str]): A dictionary where keys are old column names
            and values are new column names.

    Returns:
        DataFrame: The DataFrame with renamed columns.
    """
    for old_name, new_name in column_mapping.items():
        df = df.withColumnRenamed(old_name, new_name)
    return df


def sample_dataframe(
    df: DataFrame, fraction: float, seed: Optional[int] = None
) -> DataFrame:
    """Sample a fraction of the Snowpark DataFrame.

    Args:
        df (DataFrame): The Snowpark DataFrame to sample.
        fraction (float): The fraction of rows to sample (between 0 and 1).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        DataFrame: The sampled DataFrame.

    Raises:
        ValueError: If fraction is not between 0 and 1.
    """
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be between 0 and 1")
    return df.sample(fraction=fraction, seed=seed)


def add_row_number(
    df: DataFrame,
    partition_by: Optional[List[str]] = None,
    order_by: Optional[List[str]] = None,
    column_name: str = "row_number",
) -> DataFrame:
    """Add a row number column to the Snowpark DataFrame.

    Args:
        df (DataFrame): The Snowpark DataFrame to add row numbers to.
        partition_by (List[str], optional): Columns to partition by for windowing.
        order_by (List[str], optional): Columns to order by within partitions.
        column_name (str): The name of the new row number column.

    Returns:
        DataFrame: The DataFrame with the row number column added.
    """
    from snowflake.snowpark.functions import row_number
    from snowflake.snowpark.window import Window

    window_spec = Window.orderBy(*order_by) if order_by else Window.orderBy()
    if partition_by:
        window_spec = window_spec.partitionBy(*partition_by)

    return df.withColumn(column_name, row_number().over(window_spec))


def filter_by_condition(df: DataFrame, condition: str) -> DataFrame:
    """Filter the Snowpark DataFrame based on a SQL-like condition string.

    Args:
        df (DataFrame): The Snowpark DataFrame to filter.
        condition (str): The filter condition as a string (e.g., "age > 18").

    Returns:
        DataFrame: The filtered DataFrame.
    """
    return df.filter(condition)
