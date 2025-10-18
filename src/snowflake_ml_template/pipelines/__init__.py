"""ML Pipelines for Snowflake MLOps.

This module contains production-ready ML pipelines built on the framework.
"""

from snowflake_ml_template.pipelines._base.template import PipelineTemplate
from snowflake_ml_template.pipelines.fraud_detection import FraudDetectionPipeline
from snowflake_ml_template.pipelines.vehicle_insurance_fraud import (
    VehicleInsuranceFraudPipeline,
)

__all__ = [
    "PipelineTemplate",
    "FraudDetectionPipeline",
    "VehicleInsuranceFraudPipeline",
]
