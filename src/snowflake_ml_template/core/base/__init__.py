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
    DeploymentStrategy,
    DeploymentTarget,
)
from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    DataSource,
    IngestionConfig,
    IngestionMethod,
    IngestionResult,
    IngestionStatus,
    SourceType,
)
from snowflake_ml_template.core.base.pipeline import (
    BasePipeline,
    PipelineConfig,
    PipelineExecutionStatus,
    PipelineResult,
    PipelineStage,
)
from snowflake_ml_template.core.base.tracking import ExecutionEventTracker
from snowflake_ml_template.core.base.training import (
    BaseModelConfig,
    BaseTrainer,
    MLFramework,
    TrainingConfig,
    TrainingResult,
    TrainingStatus,
    TrainingStrategy,
)
from snowflake_ml_template.core.base.transformation import (
    BaseTransformation,
    TransformationConfig,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)

__all__ = [
    # Pipeline
    "BasePipeline",
    "PipelineConfig",
    "PipelineExecutionStatus",
    "PipelineResult",
    "PipelineStage",
    # Ingestion
    "BaseIngestionStrategy",
    "DataSource",
    "IngestionConfig",
    "IngestionResult",
    "IngestionMethod",
    "IngestionStatus",
    "SourceType",
    # Transformation
    "BaseTransformation",
    "TransformationConfig",
    "TransformationResult",
    "TransformationStatus",
    "TransformationType",
    # Training
    "BaseTrainer",
    "BaseModelConfig",
    "MLFramework",
    "TrainingConfig",
    "TrainingResult",
    "TrainingStatus",
    "TrainingStrategy",
    # Deployment
    "BaseDeploymentStrategy",
    "DeploymentConfig",
    "DeploymentResult",
    "DeploymentStrategy",
    "DeploymentTarget",
    # Tracking
    "ExecutionEventTracker",
]
