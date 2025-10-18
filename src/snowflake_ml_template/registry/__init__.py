"""Model registry for versioning, promotion, and lineage.

This module provides comprehensive model registry capabilities following
the Golden Migration Plan's three-tier promotion workflow (DEV → TEST → PROD).

Classes:
    ModelRegistry: Central model registry
    ModelVersion: Model version metadata
    ModelStage: Model deployment stages
"""

from snowflake_ml_template.registry.manager import (
    ModelRegistry,
    ModelStage,
    ModelVersion,
)

__all__ = ["ModelRegistry", "ModelVersion", "ModelStage"]
