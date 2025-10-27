"""Machine learning utilities for Snowflake ML pipelines.

This module provides functions for preprocessing, feature engineering, and model evaluation
to support ML workflows in Snowflake environments.
"""

from typing import Dict, List, Tuple, cast

import numpy as np
from snowflake.snowpark import DataFrame
from snowflake.snowpark.functions import abs, avg, col, max, min, stddev, when


def normalize_features(
    df: DataFrame, feature_columns: List[str], method: str = "standard"
) -> DataFrame:
    """Normalize numeric features using specified method.

    Args:
        df (DataFrame): The DataFrame containing features.
        feature_columns (List[str]): Columns to normalize.
        method (str): Normalization method ('standard' for z-score, 'minmax' for 0-1 scaling).

    Returns:
        DataFrame: DataFrame with normalized features.

    Raises:
        ValueError: If method is not supported.
    """
    if method not in ["standard", "minmax"]:
        raise ValueError("Method must be 'standard' or 'minmax'")

    for column in feature_columns:
        if method == "standard":
            # Calculate mean and std
            stats = (
                df.select(col(column).alias("value"))
                .agg(avg("value").alias("mean"), stddev("value").alias("std"))
                .collect()[0]
            )

            mean_val = stats["MEAN"]
            std_val = stats["STD"]

            df = df.withColumn(
                f"{column}_normalized", (col(column) - mean_val) / std_val
            )
        elif method == "minmax":
            # Calculate min and max
            stats = (
                df.select(col(column).alias("value"))
                .agg(min("value").alias("min_val"), max("value").alias("max_val"))
                .collect()[0]
            )

            min_val = stats["MIN_VAL"]
            max_val = stats["MAX_VAL"]

            df = df.withColumn(
                f"{column}_normalized", (col(column) - min_val) / (max_val - min_val)
            )

    return df


def encode_categorical_features(
    df: DataFrame, categorical_columns: List[str], method: str = "onehot"
) -> DataFrame:
    """Encode categorical features.

    Args:
        df (DataFrame): The DataFrame containing categorical features.
        categorical_columns (List[str]): Columns to encode.
        method (str): Encoding method ('onehot' or 'label').

    Returns:
        DataFrame: DataFrame with encoded features.

    Raises:
        ValueError: If method is not supported.
    """
    if method not in ["onehot", "label"]:
        raise ValueError("Method must be 'onehot' or 'label'")

    for column in categorical_columns:
        if method == "label":
            # Simple label encoding (0 to n-1)
            distinct_values = df.select(col(column)).distinct().collect()
            value_map = {row[column]: i for i, row in enumerate(distinct_values)}

            # Create a case statement for encoding
            case_expr = None
            for value, code in value_map.items():
                if case_expr is None:
                    case_expr = when(col(column) == value, code)
                else:
                    case_expr = case_expr.when(col(column) == value, code)
            if case_expr is not None:
                case_expr = case_expr.otherwise(-1)  # Handle unseen values
            else:
                case_expr = when(
                    col(column).is_null(), -1
                )  # If no values, handle nulls

            df = df.withColumn(f"{column}_encoded", case_expr)
        elif method == "onehot":
            # One-hot encoding (simplified - in practice, might need more complex logic)
            distinct_values = df.select(col(column)).distinct().collect()
            categories = [row[column] for row in distinct_values]

            for category in categories:
                df = df.withColumn(
                    f"{column}_{category}",
                    when(col(column) == category, 1).otherwise(0),
                )

    return df


def split_train_test(
    df: DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """Split DataFrame into train and test sets.

    Args:
        df (DataFrame): The DataFrame to split.
        target_column (str): The target column for stratified sampling.
        test_size (float): Proportion of data for testing.
        random_state (int): Random seed.

    Returns:
        Tuple[DataFrame, DataFrame]: (train_df, test_df)
    """
    # Simple random split (in practice, might want stratified sampling)
    test_df = df.sample(fraction=test_size, seed=random_state)

    # Get remaining for train
    train_df = df.subtract(test_df)

    return train_df, test_df


def calculate_metrics(
    predictions_df: DataFrame,
    actual_col: str,
    predicted_col: str,
    task: str = "classification",
) -> Dict[str, float]:
    """Calculate evaluation metrics for predictions.

    Args:
        predictions_df (DataFrame): DataFrame with actual and predicted values.
        actual_col (str): Column name for actual values.
        predicted_col (str): Column name for predicted values.
        task (str): 'classification' or 'regression'.

    Returns:
        Dict[str, float]: Dictionary of calculated metrics.
    """
    if task == "regression":
        # Calculate RMSE, MAE, R2
        metrics = (
            predictions_df.select(
                ((col(actual_col) - col(predicted_col)) ** 2).alias("squared_error"),
                abs(col(actual_col) - col(predicted_col)).alias("abs_error"),
                col(actual_col).alias("actual"),
                col(predicted_col).alias("predicted"),
            )
            .agg(avg("squared_error").alias("mse"), avg("abs_error").alias("mae"))
            .collect()[0]
        )
        mse = metrics["MSE"]
        rmse = np.sqrt(mse)
        mae = metrics["MAE"]

        # R2 calculation would require more complex SQL
        return {"rmse": rmse, "mae": mae}

    elif task == "classification":
        # Calculate accuracy, precision, recall (simplified)
        correct_predictions = predictions_df.filter(
            col(actual_col) == col(predicted_col)
        ).count()
        total_predictions = predictions_df.count()
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )

        return {"accuracy": accuracy}

    else:
        raise ValueError("Task must be 'classification' or 'regression'")


def handle_class_imbalance(
    df: DataFrame, target_column: str, method: str = "undersample"
) -> DataFrame:
    """Handle class imbalance in classification datasets.

    Args:
        df (DataFrame): The DataFrame with imbalanced classes.
        target_column (str): The target column.
        method (str): Balancing method ('undersample' or 'oversample').

    Returns:
        DataFrame: Balanced DataFrame.

    Raises:
        ValueError: If method is not supported.
    """
    if method not in ["undersample", "oversample"]:
        raise ValueError("Method must be 'undersample' or 'oversample'")

    # Get class counts
    class_counts = df.groupBy(target_column).count().collect()
    class_counts = {row[target_column]: row["COUNT"] for row in class_counts}

    import builtins

    min_class = builtins.min(class_counts.keys(), key=lambda k: class_counts[k])
    max_class = builtins.max(class_counts.keys(), key=lambda k: class_counts[k])
    min_count = class_counts[min_class]

    if method == "undersample":
        # Undersample majority class
        undersampled_df = df.filter(col(target_column) == min_class)
        for cls in class_counts:
            if cls != min_class:
                cls_sample = df.filter(col(target_column) == cls).sample(
                    fraction=min_count / class_counts[cls]
                )
                undersampled_df = undersampled_df.union(cls_sample)
        return cast(DataFrame, undersampled_df)

    elif method == "oversample":
        # Oversample minority class (simplified - in practice, use SMOTE or similar)
        oversampled_df = df
        for cls in class_counts:
            if cls != max_class:
                cls_df = df.filter(col(target_column) == cls)
                # Duplicate minority class samples
                multiplier = int(class_counts[max_class] / class_counts[cls])
                for _ in range(multiplier - 1):
                    oversampled_df = oversampled_df.union(cls_df)
        return oversampled_df

    return df
