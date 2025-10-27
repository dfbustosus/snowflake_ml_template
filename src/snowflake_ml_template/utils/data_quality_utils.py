"""Data quality utilities for Snowflake ML pipelines.

This module provides functions for data profiling, validation, and quality checks
to ensure data integrity in ML workflows.
"""

from typing import Any, Dict, List, Tuple, cast

from snowflake.snowpark import DataFrame
from snowflake.snowpark.functions import (
    approx_percentile,
    col,
    count,
    count_distinct,
    is_null,
    max,
    mean,
    min,
    stddev,
)


def profile_dataframe(df: DataFrame) -> Dict[str, Dict[str, Any]]:
    """Generate a data profile for the DataFrame.

    Args:
        df (DataFrame): The DataFrame to profile.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary with column profiles containing
        statistics like count, distinct count, null count, mean, std, min, max.
    """
    profile = {}
    columns = [field.name for field in df.schema.fields]

    for column in columns:
        col_stats = df.select(
            count(col(column)).alias("total_count"),
            count_distinct(col(column)).alias("distinct_count"),
            count(is_null(col(column))).alias("null_count"),
            mean(col(column)).alias("mean"),
            stddev(col(column)).alias("stddev"),
            min(col(column)).alias("min"),
            max(col(column)).alias("max"),
        ).collect()[0]

        profile[column] = {
            "total_count": col_stats["TOTAL_COUNT"],
            "distinct_count": col_stats["DISTINCT_COUNT"],
            "null_count": col_stats["NULL_COUNT"],
            "null_percentage": (
                (col_stats["NULL_COUNT"] / col_stats["TOTAL_COUNT"]) * 100
                if col_stats["TOTAL_COUNT"] > 0
                else 0
            ),
            "mean": col_stats["MEAN"],
            "stddev": col_stats["STDDEV"],
            "min": col_stats["MIN"],
            "max": col_stats["MAX"],
        }

    return profile


def detect_outliers(
    df: DataFrame, column: str, method: str = "iqr", threshold: float = 1.5
) -> DataFrame:
    """Detect outliers in a numeric column using specified method.

    Args:
        df (DataFrame): The DataFrame to analyze.
        column (str): The column to check for outliers.
        method (str): Method to use ('iqr' for interquartile range).
        threshold (float): Threshold for outlier detection.

    Returns:
        DataFrame: DataFrame with an additional 'is_outlier' column.
    """
    if method == "iqr":
        # Calculate Q1, Q3, IQR
        quantiles = df.select(
            approx_percentile(col(column), 0.25).alias("q1"),
            approx_percentile(col(column), 0.75).alias("q3"),
        ).collect()[0]
        q1 = quantiles["Q1"]
        q3 = quantiles["Q3"]
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        return cast(
            DataFrame,
            df.withColumn(
                "is_outlier", (col(column) < lower_bound) | (col(column) > upper_bound)
            ),
        )

    raise ValueError(f"Unsupported outlier detection method: {method}")


def check_data_completeness(
    df: DataFrame, threshold: float = 0.95
) -> Tuple[bool, Dict[str, float]]:
    """Check if the DataFrame meets completeness requirements.

    Args:
        df (DataFrame): The DataFrame to check.
        threshold (float): Minimum completeness ratio required (0-1).

    Returns:
        Tuple[bool, Dict[str, float]]: (is_complete, completeness_ratios)
    """
    columns = [field.name for field in df.schema.fields]
    completeness_ratios = {}

    for column in columns:
        total_count = df.count()
        non_null_count = df.filter(col(column).is_not_null()).count()
        ratio = non_null_count / total_count if total_count > 0 else 0
        completeness_ratios[column] = ratio

    is_complete = all(ratio >= threshold for ratio in completeness_ratios.values())
    return is_complete, completeness_ratios


def validate_data_types(
    df: DataFrame, expected_types: Dict[str, str]
) -> Tuple[bool, List[str]]:
    """Validate that columns have expected data types.

    Args:
        df (DataFrame): The DataFrame to validate.
        expected_types (Dict[str, str]): Expected types for columns.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation errors)
    """
    errors = []
    actual_schema = {
        field.name: field.datatype.__class__.__name__.replace("Type", "").upper()
        for field in df.schema.fields
    }

    for column, expected_type in expected_types.items():
        if column not in actual_schema:
            errors.append(f"Column '{column}' is missing")
        elif actual_schema[column] != expected_type.upper():
            errors.append(
                f"Column '{column}' has type '{actual_schema[column]}', expected '{expected_type.upper()}'"
            )

    return len(errors) == 0, errors
