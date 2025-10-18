"""Base pipeline template for all ML pipelines.

This module provides the concrete implementation of BasePipeline,
serving as a template for all model-specific pipelines.

Classes:
    PipelineTemplate: Concrete pipeline implementation
"""

from snowflake_ml_template.pipelines._base.template import PipelineTemplate

__all__ = ["PipelineTemplate"]
