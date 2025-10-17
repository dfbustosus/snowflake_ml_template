"""Abstract base classes and interfaces for the MLOps framework.

This module defines the core abstractions that all concrete implementations
must follow. By depending on these abstractions rather than concrete classes,
we achieve loose coupling and high testability.

Classes:
    BasePipeline: Abstract base class for ML pipelines
    BaseIngestionStrategy: Interface for data ingestion strategies
    BaseTransformation: Interface for data transformations
    BaseTrainer: Interface for model training strategies
    BaseDeploymentStrategy: Interface for model deployment strategies
    PipelineStage: Enum defining pipeline execution stages
    PipelineConfig: Configuration dataclass for pipelines
    PipelineResult: Result dataclass for pipeline execution
"""

from snowflake_ml_template.core.base.deployment import (
    BaseDeploymentStrategy,
    DeploymentConfig,
    DeploymentResult,
)
from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    DataSource,
    IngestionConfig,
    IngestionResult,
)
from snowflake_ml_template.core.base.pipeline import (
    BasePipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)
from snowflake_ml_template.core.base.training import (
    BaseModelConfig,
    BaseTrainer,
    TrainingConfig,
    TrainingResult,
)
from snowflake_ml_template.core.base.transformation import (
    BaseTransformation,
    TransformationConfig,
    TransformationResult,
)

__all__ = [
    # Pipeline
    "BasePipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    # Ingestion
    "BaseIngestionStrategy",
    "DataSource",
    "IngestionConfig",
    "IngestionResult",
    # Transformation
    "BaseTransformation",
    "TransformationConfig",
    "TransformationResult",
    # Training
    "BaseTrainer",
    "BaseModelConfig",
    "TrainingConfig",
    "TrainingResult",
    # Deployment
    "BaseDeploymentStrategy",
    "DeploymentConfig",
    "DeploymentResult",
]
