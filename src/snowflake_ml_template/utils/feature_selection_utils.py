"""Feature selection utilities for Snowflake ML pipelines.

This module provides functions for selecting optimal features to improve model performance,
reduce dimensionality, and enhance interpretability in ML workflows.
"""

from typing import List, Optional

from snowflake.snowpark import DataFrame
from snowflake.snowpark.functions import col, corr, var_samp


def select_features_by_correlation(
    df: DataFrame, target_column: str, threshold: float = 0.8
) -> List[str]:
    """Select features based on correlation with target and multicollinearity.

    Args:
        df (DataFrame): The DataFrame containing features and target.
        target_column (str): The target column name.
        threshold (float): Correlation threshold for feature selection.

    Returns:
        List[str]: List of selected feature column names.
    """
    numeric_columns = [
        field.name
        for field in df.schema.fields
        if field.datatype.__class__.__name__
        in ["IntegerType", "LongType", "FloatType", "DoubleType"]
        and field.name != target_column
    ]

    selected_features = []

    for feature in numeric_columns:
        # Calculate correlation with target
        corr_with_target = df.select(
            corr(col(target_column), col(feature)).alias("correlation")
        ).collect()[0]["CORRELATION"]

        if abs(corr_with_target) > threshold:
            selected_features.append(feature)

    return selected_features


def select_features_by_variance(
    df: DataFrame, threshold: float = 0.01, exclude_columns: Optional[List[str]] = None
) -> List[str]:
    """Select features based on variance threshold.

    Args:
        df (DataFrame): The DataFrame containing features.
        threshold (float): Minimum variance threshold.
        exclude_columns (List[str]): Columns to exclude from selection.

    Returns:
        List[str]: List of selected feature column names.
    """
    if exclude_columns is None:
        exclude_columns = []

    numeric_columns = [
        field.name
        for field in df.schema.fields
        if field.datatype.__class__.__name__
        in ["IntegerType", "LongType", "FloatType", "DoubleType"]
        and field.name not in exclude_columns
    ]

    selected_features = []

    for feature in numeric_columns:
        variance = df.select(var_samp(col(feature)).alias("variance")).collect()[0][
            "VARIANCE"
        ]
        if variance > threshold:
            selected_features.append(feature)

    return selected_features


def remove_multicollinear_features(
    df: DataFrame, threshold: float = 0.9, exclude_columns: Optional[List[str]] = None
) -> List[str]:
    """Remove highly correlated features to reduce multicollinearity.

    Args:
        df (DataFrame): The DataFrame containing features.
        threshold (float): Correlation threshold for removal.
        exclude_columns (List[str]): Columns to exclude from analysis.

    Returns:
        List[str]: List of selected feature column names.
    """
    if exclude_columns is None:
        exclude_columns = []

    numeric_columns = [
        field.name
        for field in df.schema.fields
        if field.datatype.__class__.__name__
        in ["IntegerType", "LongType", "FloatType", "DoubleType"]
        and field.name not in exclude_columns
    ]

    to_remove = set()

    for i, col1 in enumerate(numeric_columns):
        for col2 in numeric_columns[i + 1 :]:
            if col1 in to_remove or col2 in to_remove:
                continue

            correlation = df.select(corr(col(col1), col(col2)).alias("corr")).collect()[
                0
            ]["CORR"]
            if abs(correlation) > threshold:
                to_remove.add(col2)  # Remove the second one

    return [col for col in numeric_columns if col not in to_remove]


def select_features_by_missing_values(
    df: DataFrame, threshold: float = 0.5, exclude_columns: Optional[List[str]] = None
) -> List[str]:
    """Select features based on missing value ratio.

    Args:
        df (DataFrame): The DataFrame containing features.
        threshold (float): Maximum allowed missing value ratio.
        exclude_columns (List[str]): Columns to exclude from selection.

    Returns:
        List[str]: List of selected feature column names.
    """
    if exclude_columns is None:
        exclude_columns = []

    all_columns = [
        field.name for field in df.schema.fields if field.name not in exclude_columns
    ]

    selected_features = []

    for column in all_columns:
        total_count = df.count()
        non_null_count = df.filter(col(column).is_not_null()).count()
        missing_ratio = 1 - (non_null_count / total_count) if total_count > 0 else 1

        if missing_ratio <= threshold:
            selected_features.append(column)

    return selected_features
