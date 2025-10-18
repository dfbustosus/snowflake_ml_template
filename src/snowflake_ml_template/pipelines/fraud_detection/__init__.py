"""Fraud detection reference pipeline.

This pipeline demonstrates the complete MLOps workflow for fraud detection,
including feature engineering, model training, and deployment.
"""

from snowflake_ml_template.pipelines.fraud_detection.pipeline import (
    FraudDetectionPipeline,
)

__all__ = ["FraudDetectionPipeline"]
