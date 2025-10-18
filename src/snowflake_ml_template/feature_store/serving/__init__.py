"""Feature serving for training and inference.

This module provides feature serving capabilities with point-in-time
correctness guarantees, ensuring no data leakage in training datasets.

Classes:
    BatchFeatureServer: Batch feature serving with ASOF joins
    OnlineFeatureServer: Online feature serving (future)
"""

from snowflake_ml_template.feature_store.serving.batch import BatchFeatureServer

__all__ = ["BatchFeatureServer"]
