"""
Unit tests for ml_utils.py.

These tests validate the functionality of ML utilities.
"""

from unittest.mock import MagicMock

import pytest
from snowflake.snowpark import Row

from src.snowflake_ml_template.utils.ml_utils import (
    calculate_metrics,
    encode_categorical_features,
    handle_class_imbalance,
    normalize_features,
    split_train_test,
)


@pytest.fixture(scope="module")
def snowpark_session():
    """Fixture to create a mocked Snowpark session for testing."""
    session = MagicMock()
    session.create_dataframe = MagicMock()
    return session


def test_normalize_features_standard(snowpark_session):
    """Test normalize_features with standard method."""
    df = snowpark_session.create_dataframe.return_value
    df.select.return_value.agg.return_value.collect.return_value = [
        Row(MEAN=30.0, STD=5.0)
    ]
    df.withColumn.return_value = df
    normalized_df = normalize_features(df, ["age"], method="standard")
    assert normalized_df == df


def test_normalize_features_minmax(snowpark_session):
    """Test normalize_features with minmax method."""
    df = snowpark_session.create_dataframe.return_value
    df.select.return_value.agg.return_value.collect.return_value = [
        Row(MIN_VAL=20, MAX_VAL=40)
    ]
    df.withColumn.return_value = df
    normalized_df = normalize_features(df, ["age"], method="minmax")
    assert normalized_df == df


def test_encode_categorical_features_label(snowpark_session):
    """Test encode_categorical_features with label method."""
    df = snowpark_session.create_dataframe.return_value
    df.select.return_value.distinct.return_value.collect.return_value = [
        Row(category="A"),
        Row(category="B"),
    ]
    df.withColumn.return_value = df
    encoded_df = encode_categorical_features(df, ["category"], method="label")
    assert encoded_df == df


def test_encode_categorical_features_onehot(snowpark_session):
    """Test encode_categorical_features with onehot method."""
    df = snowpark_session.create_dataframe.return_value
    df.select.return_value.distinct.return_value.collect.return_value = [
        Row(category="A"),
        Row(category="B"),
    ]
    df.withColumn.return_value = df
    encoded_df = encode_categorical_features(df, ["category"], method="onehot")
    assert encoded_df == df


def test_split_train_test(snowpark_session):
    """Test split_train_test function."""
    df = snowpark_session.create_dataframe.return_value
    df.count.return_value = 100
    df.sample.return_value = df
    df.subtract.return_value = df
    train_df, test_df = split_train_test(df, "target")
    assert train_df == df
    assert test_df == df


def test_calculate_metrics_regression(snowpark_session):
    """Test calculate_metrics for regression."""
    df = snowpark_session.create_dataframe.return_value
    df.select.return_value.agg.return_value.collect.return_value = [
        Row(MSE=4.0, MAE=2.0)
    ]
    metrics = calculate_metrics(df, "actual", "predicted", task="regression")
    assert "rmse" in metrics
    assert "mae" in metrics


def test_calculate_metrics_classification(snowpark_session):
    """Test calculate_metrics for classification."""
    df = snowpark_session.create_dataframe.return_value
    df.filter.return_value.count.return_value = 80
    df.count.return_value = 100
    metrics = calculate_metrics(df, "actual", "predicted", task="classification")
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.8


def test_handle_class_imbalance_undersample(snowpark_session):
    """Test handle_class_imbalance with undersample method."""
    df = snowpark_session.create_dataframe.return_value
    df.groupBy.return_value.count.return_value.collect.return_value = [
        Row(target=0, COUNT=50),
        Row(target=1, COUNT=10),
    ]
    df.filter.return_value = df
    df.sample.return_value = df
    df.union.return_value = df
    balanced_df = handle_class_imbalance(df, "target", method="undersample")
    assert balanced_df == df
