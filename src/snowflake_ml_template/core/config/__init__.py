"""Configuration management for the MLOps framework.

This module provides type-safe configuration loading and validation using
Pydantic models. It supports loading from YAML files, environment variables,
and hierarchical configuration merging.

Classes:
    ConfigLoader: Load and merge configurations from multiple sources
    ConfigValidator: Validate configuration completeness and correctness

Pydantic Models:
    SnowflakeConfig: Snowflake connection configuration
    FeatureStoreConfig: Feature store configuration
    ModelRegistryConfig: Model registry configuration
    MonitoringConfig: Monitoring configuration
"""

from snowflake_ml_template.core.config.loader import ConfigLoader
from snowflake_ml_template.core.config.models import (
    DeploymentConfigModel,
    FeatureStoreConfig,
    IngestionConfigModel,
    ModelRegistryConfig,
    MonitoringConfig,
    SnowflakeConfig,
    TrainingConfigModel,
    TransformationConfigModel,
)
from snowflake_ml_template.core.config.validator import ConfigValidator

__all__ = [
    "ConfigLoader",
    "ConfigValidator",
    "SnowflakeConfig",
    "FeatureStoreConfig",
    "ModelRegistryConfig",
    "MonitoringConfig",
    "IngestionConfigModel",
    "TransformationConfigModel",
    "TrainingConfigModel",
    "DeploymentConfigModel",
]
