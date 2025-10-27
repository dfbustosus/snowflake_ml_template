"""Model management utilities for Snowflake ML pipelines.

This module provides functions for model persistence, loading, and advanced evaluation
metrics to support production ML workflows.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from snowflake.snowpark import DataFrame
from snowflake.snowpark.functions import abs, col


def save_model_metadata(
    model_name: str,
    model_type: str,
    features: List[str],
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    database: str,
    schema: str,
    table: str,
    session: Any,
) -> None:
    """Save model metadata to a Snowflake table.

    Args:
        model_name (str): Name of the model.
        model_type (str): Type of model (e.g., 'linear_regression').
        features (List[str]): List of feature names used.
        hyperparameters (Dict[str, Any]): Model hyperparameters.
        metrics (Dict[str, float]): Evaluation metrics.
        database (str): Snowflake database name.
        schema (str): Snowflake schema name.
        table (str): Table name for storing metadata.
        session: Snowpark session.
    """
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "features": json.dumps(features),
        "hyperparameters": json.dumps(hyperparameters),
        "metrics": json.dumps(metrics),
        "created_at": "CURRENT_TIMESTAMP",
    }

    # Create table if not exists
    session.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {database}.{schema}.{table} (
            model_name STRING,
            model_type STRING,
            features STRING,
            hyperparameters STRING,
            metrics STRING,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    ).collect()

    # Insert metadata
    session.sql(
        f"""
        INSERT INTO {database}.{schema}.{table} (model_name, model_type, features, hyperparameters, metrics)
        SELECT
            '{metadata["model_name"]}',
            '{metadata["model_type"]}',
            '{metadata["features"]}',
            '{metadata["hyperparameters"]}',
            '{metadata["metrics"]}'
    """
    ).collect()


def load_model_metadata(
    model_name: str, database: str, schema: str, table: str, session: Any
) -> Dict[str, Any]:
    """Load model metadata from Snowflake table.

    Args:
        model_name (str): Name of the model.
        database (str): Snowflake database name.
        schema (str): Snowflake schema name.
        table (str): Table name storing metadata.
        session: Snowpark session.

    Returns:
        Dict[str, Any]: Model metadata.
    """
    result = session.sql(
        f"""
        SELECT * FROM {database}.{schema}.{table}
        WHERE model_name = '{model_name}'
        ORDER BY created_at DESC
        LIMIT 1
    """
    ).collect()

    if not result:
        raise ValueError(f"Model {model_name} not found")

    row = result[0]
    return {
        "model_name": row["MODEL_NAME"],
        "model_type": row["MODEL_TYPE"],
        "features": json.loads(row["FEATURES"]),
        "hyperparameters": json.loads(row["HYPERPARAMETERS"]),
        "metrics": json.loads(row["METRICS"]),
        "created_at": row["CREATED_AT"],
    }


def calculate_classification_metrics(
    predictions_df: DataFrame,
    actual_col: str,
    predicted_col: str,
    predicted_prob_col: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics.

    Args:
        predictions_df (DataFrame): DataFrame with actual and predicted values.
        actual_col (str): Column name for actual values.
        predicted_col (str): Column name for predicted values.
        predicted_prob_col (str): Column name for predicted probabilities (optional).

    Returns:
        Dict[str, float]: Dictionary of calculated metrics.
    """
    # Basic metrics
    total_count = predictions_df.count()
    correct_count = predictions_df.filter(col(actual_col) == col(predicted_col)).count()
    accuracy = correct_count / total_count if total_count > 0 else 0

    # Confusion matrix components
    tp = predictions_df.filter(
        (col(actual_col) == 1) & (col(predicted_col) == 1)
    ).count()
    tn = predictions_df.filter(
        (col(actual_col) == 0) & (col(predicted_col) == 0)
    ).count()
    fp = predictions_df.filter(
        (col(actual_col) == 0) & (col(predicted_col) == 1)
    ).count()
    fn = predictions_df.filter(
        (col(actual_col) == 1) & (col(predicted_col) == 0)
    ).count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }

    # Add AUC if probabilities are available
    if predicted_prob_col:
        # Simplified AUC calculation (in practice, use proper implementation)
        metrics[
            "auc"
        ] = 0.5  # Placeholder - proper AUC requires sorting and calculation

    return metrics


def calculate_regression_metrics(
    predictions_df: DataFrame, actual_col: str, predicted_col: str
) -> Dict[str, float]:
    """Calculate comprehensive regression metrics.

    Args:
        predictions_df (DataFrame): DataFrame with actual and predicted values.
        actual_col (str): Column name for actual values.
        predicted_col (str): Column name for predicted values.

    Returns:
        Dict[str, float]: Dictionary of calculated metrics.
    """
    # Calculate metrics using SQL aggregations
    metrics_df = (
        predictions_df.select(
            ((col(actual_col) - col(predicted_col)) ** 2).alias("squared_error"),
            abs(col(actual_col) - col(predicted_col)).alias("abs_error"),
            col(actual_col).alias("actual"),
            col(predicted_col).alias("predicted"),
        )
        .agg(
            "count(*)",
            "avg(squared_error)",
            "avg(abs_error)",
            "avg(actual)",
            "avg(predicted)",
        )
        .collect()[0]
    )

    n = metrics_df["COUNT(*)"]
    mse = metrics_df["AVG(SQUARED_ERROR)"]
    mae = metrics_df["AVG(ABS_ERROR)"]
    mean_actual = metrics_df["AVG(ACTUAL)"]

    rmse = np.sqrt(mse)

    # R-squared calculation
    ss_res = mse * n
    ss_tot = sum(
        (row["ACTUAL"] - mean_actual) ** 2
        for row in predictions_df.select(col(actual_col)).collect()
    )
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {"mse": mse, "rmse": rmse, "mae": mae, "r_squared": r_squared}


def validate_model_predictions(
    predictions_df: DataFrame,
    actual_col: str,
    predicted_col: str,
    task: str = "classification",
) -> Dict[str, Any]:
    """Validate model predictions and return comprehensive assessment.

    Args:
        predictions_df (DataFrame): DataFrame with predictions.
        actual_col (str): Actual values column.
        predicted_col (str): Predicted values column.
        task (str): 'classification' or 'regression'.

    Returns:
        Dict[str, Any]: Validation results including metrics and data quality checks.
    """
    # Data quality checks
    total_predictions = predictions_df.count()
    null_actual = predictions_df.filter(col(actual_col).is_null()).count()
    null_predicted = predictions_df.filter(col(predicted_col).is_null()).count()

    validation_results = {
        "total_predictions": total_predictions,
        "null_actual_count": null_actual,
        "null_predicted_count": null_predicted,
        "data_quality_issues": [],
    }

    if null_actual > 0:
        validation_results["data_quality_issues"].append(
            f"{null_actual} null values in actual column"
        )

    if null_predicted > 0:
        validation_results["data_quality_issues"].append(
            f"{null_predicted} null values in predicted column"
        )

    # Calculate metrics
    if task == "classification":
        metrics = calculate_classification_metrics(
            predictions_df, actual_col, predicted_col
        )
    elif task == "regression":
        metrics = calculate_regression_metrics(
            predictions_df, actual_col, predicted_col
        )
    else:
        raise ValueError("Task must be 'classification' or 'regression'")

    validation_results["metrics"] = metrics

    return validation_results
