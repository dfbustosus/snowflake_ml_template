"""Custom exception classes for the MLOps framework.

This module defines all custom exceptions used throughout the framework.
Each exception includes detailed error messages and optional context
information for debugging.

All exceptions inherit from MLOpsError, making it easy to catch all
framework-specific errors with a single except clause.
"""

from typing import Any, Dict, Optional


class MLOpsError(Exception):
    """Base exception for all MLOps framework errors.

    This is the base class for all custom exceptions in the framework.
    It provides a consistent interface for error handling and includes
    optional context information for debugging.

    Attributes:
        message: Error message
        context: Optional dictionary with additional context
        original_error: Optional original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            context: Optional context dictionary
            original_error: Optional original exception
        """
        self.message = message
        self.context = context or {}
        self.original_error = original_error

        # Build full error message
        full_message = message
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message = f"{message} (Context: {context_str})"
        if original_error:
            full_message = f"{full_message} (Caused by: {str(original_error)})"

        super().__init__(full_message)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(MLOpsError):
    """Exception raised for configuration errors.

    This exception is raised when there are issues with configuration
    loading, validation, or when required configuration is missing.
    """

    pass


# =============================================================================
# Session Errors
# =============================================================================


class SessionError(MLOpsError):
    """Exception raised for Snowflake session errors.

    This exception is raised when there are issues with session creation,
    management, or when session operations fail.
    """

    pass


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(MLOpsError):
    """Base exception for pipeline errors.

    This is the base class for all pipeline-related errors.
    """

    pass


class PipelineValidationError(PipelineError):
    """Exception raised when pipeline validation fails.

    This exception is raised when pipeline configuration or prerequisites
    fail validation before execution.
    """

    pass


class PipelineExecutionError(PipelineError):
    """Exception raised when pipeline execution fails.

    This exception is raised when a pipeline fails during execution,
    typically wrapping the underlying error with additional context.
    """

    pass


# =============================================================================
# Ingestion Errors
# =============================================================================


class IngestionError(MLOpsError):
    """Base exception for data ingestion errors.

    This is the base class for all ingestion-related errors.
    """

    pass


class SourceConnectionError(IngestionError):
    """Exception raised when connection to data source fails.

    This exception is raised when the framework cannot connect to
    a data source (S3, Azure, GCS, etc.).
    """

    pass


class DataValidationError(IngestionError):
    """Exception raised when data validation fails.

    This exception is raised when ingested data fails validation
    checks (schema, quality, etc.).
    """

    pass


# =============================================================================
# Transformation Errors
# =============================================================================


class TransformationError(MLOpsError):
    """Exception raised for data transformation errors.

    This exception is raised when data transformations fail,
    whether using Snowpark, SQL, or dbt.
    """

    pass


# =============================================================================
# Feature Store Errors
# =============================================================================


class FeatureStoreError(MLOpsError):
    """Base exception for feature store errors.

    This is the base class for all feature store-related errors.
    """

    pass


class EntityNotFoundError(FeatureStoreError):
    """Exception raised when an entity is not found.

    This exception is raised when attempting to access an entity
    that doesn't exist in the feature store.
    """

    def __init__(
        self,
        entity_name: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            entity_name: Name of the entity that was not found
            context: Optional context dictionary
            original_error: Optional original exception
        """
        message = f"Entity not found: {entity_name}"
        super().__init__(message, context, original_error)
        self.entity_name = entity_name


class FeatureViewNotFoundError(FeatureStoreError):
    """Exception raised when a feature view is not found.

    This exception is raised when attempting to access a feature view
    that doesn't exist in the feature store.
    """

    def __init__(
        self,
        feature_view_name: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            feature_view_name: Name of the feature view that was not found
            context: Optional context dictionary
            original_error: Optional original exception
        """
        message = f"Feature view not found: {feature_view_name}"
        super().__init__(message, context, original_error)
        self.feature_view_name = feature_view_name


class FeatureVersionError(FeatureStoreError):
    """Exception raised for feature versioning errors.

    This exception is raised when there are issues with feature
    versioning, such as version conflicts or invalid versions.
    """

    pass


# =============================================================================
# Training Errors
# =============================================================================


class TrainingError(MLOpsError):
    """Base exception for model training errors.

    This is the base class for all training-related errors.
    """

    pass


class DataPreparationError(TrainingError):
    """Exception raised when data preparation for training fails.

    This exception is raised when there are issues preparing
    training data (splits, feature engineering, etc.).
    """

    pass


class ModelTrainingError(TrainingError):
    """Exception raised when model training fails.

    This exception is raised when the actual model training
    process fails.
    """

    pass


# =============================================================================
# Registry Errors
# =============================================================================


class RegistryError(MLOpsError):
    """Base exception for model registry errors.

    This is the base class for all registry-related errors.
    """

    pass


class ModelNotFoundError(RegistryError):
    """Exception raised when a model is not found in the registry.

    This exception is raised when attempting to access a model
    that doesn't exist in the registry.
    """

    def __init__(
        self,
        model_name: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            model_name: Name of the model that was not found
            context: Optional context dictionary
            original_error: Optional original exception
        """
        message = f"Model not found: {model_name}"
        super().__init__(message, context, original_error)
        self.model_name = model_name


class VersionNotFoundError(RegistryError):
    """Exception raised when a model version is not found.

    This exception is raised when attempting to access a model version
    that doesn't exist in the registry.
    """

    def __init__(
        self,
        model_name: str,
        version: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            model_name: Name of the model
            version: Version that was not found
            context: Optional context dictionary
            original_error: Optional original exception
        """
        message = f"Version not found: {model_name} v{version}"
        super().__init__(message, context, original_error)
        self.model_name = model_name
        self.version = version


# =============================================================================
# Deployment Errors
# =============================================================================


class DeploymentError(MLOpsError):
    """Base exception for model deployment errors.

    This is the base class for all deployment-related errors.
    """

    pass


class DeploymentValidationError(DeploymentError):
    """Exception raised when deployment validation fails.

    This exception is raised when deployment configuration or
    prerequisites fail validation.
    """

    pass


class ServiceUnavailableError(DeploymentError):
    """Exception raised when a deployed service is unavailable.

    This exception is raised when attempting to access a deployed
    model/service that is not available or not responding.
    """

    pass


# =============================================================================
# Monitoring Errors
# =============================================================================


class MonitoringError(MLOpsError):
    """Base exception for monitoring errors.

    This is the base class for all monitoring-related errors.
    """

    pass


class DriftDetectionError(MonitoringError):
    """Exception raised when drift detection fails.

    This exception is raised when there are issues with detecting
    data or model drift.
    """

    pass


class AlertingError(MonitoringError):
    """Exception raised when alerting fails.

    This exception is raised when there are issues sending alerts
    or notifications.
    """

    pass
