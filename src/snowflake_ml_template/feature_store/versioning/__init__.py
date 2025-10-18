"""Feature versioning and lineage tracking.

This module provides comprehensive version management and lineage tracking
for feature store components, ensuring reproducibility and governance.

Classes:
    FeatureVersionManager: Manage feature view versions
    LineageTracker: Track data lineage from source to features
"""

from snowflake_ml_template.feature_store.versioning.lineage import LineageTracker
from snowflake_ml_template.feature_store.versioning.manager import FeatureVersionManager

__all__ = ["FeatureVersionManager", "LineageTracker"]
