"""Custom exception hierarchy for the MLOps framework.

This module defines a comprehensive exception hierarchy for handling
errors in the MLOps framework. All custom exceptions inherit from
MLOpsError for easy catching of framework-specific errors.

Exception Hierarchy:
    MLOpsError (base)
    ├── ConfigurationError
    ├── SessionError
    ├── PipelineError
    │   ├── PipelineValidationError
    │   └── PipelineExecutionError
    ├── IngestionError
    │   ├── SourceConnectionError
    │   └── DataValidationError
    ├── TransformationError
    ├── FeatureStoreError
    │   ├── EntityNotFoundError
    │   ├── FeatureViewNotFoundError
    │   └── FeatureVersionError
    ├── TrainingError
    │   ├── DataPreparationError
    │   └── ModelTrainingError
    ├── RegistryError
    │   ├── ModelNotFoundError
    │   └── VersionNotFoundError
    ├── DeploymentError
    │   ├── DeploymentValidationError
    │   └── ServiceUnavailableError
    └── MonitoringError
        ├── DriftDetectionError
        └── AlertingError
"""

from snowflake_ml_template.core.exceptions.errors import (  # Base exception; Configuration errors; Session errors; Pipeline errors; Ingestion errors; Transformation errors; Feature store errors; Training errors; Registry errors; Deployment errors; Monitoring errors
    AlertingError,
    ConfigurationError,
    DataPreparationError,
    DataValidationError,
    DeploymentError,
    DeploymentValidationError,
    DriftDetectionError,
    EntityNotFoundError,
    FeatureStoreError,
    FeatureVersionError,
    FeatureViewNotFoundError,
    IngestionError,
    MLOpsError,
    ModelNotFoundError,
    ModelTrainingError,
    MonitoringError,
    PipelineError,
    PipelineExecutionError,
    PipelineValidationError,
    RegistryError,
    ServiceUnavailableError,
    SessionError,
    SourceConnectionError,
    TrainingError,
    TransformationError,
    VersionNotFoundError,
)

__all__ = [
    # Base
    "MLOpsError",
    # Configuration
    "ConfigurationError",
    # Session
    "SessionError",
    # Pipeline
    "PipelineError",
    "PipelineValidationError",
    "PipelineExecutionError",
    # Ingestion
    "IngestionError",
    "SourceConnectionError",
    "DataValidationError",
    # Transformation
    "TransformationError",
    # Feature Store
    "FeatureStoreError",
    "EntityNotFoundError",
    "FeatureViewNotFoundError",
    "FeatureVersionError",
    # Training
    "TrainingError",
    "DataPreparationError",
    "ModelTrainingError",
    # Registry
    "RegistryError",
    "ModelNotFoundError",
    "VersionNotFoundError",
    # Deployment
    "DeploymentError",
    "DeploymentValidationError",
    "ServiceUnavailableError",
    # Monitoring
    "MonitoringError",
    "DriftDetectionError",
    "AlertingError",
]
