"""Enterprise Feature Store for Snowflake ML.

This module provides a comprehensive, production-ready Feature Store that follows
Snowflake's native Feature Store patterns and best practices. It supports both
Snowflake-managed and external FeatureViews, versioning, point-in-time correctness,
and automated lineage tracking.

Key Components:
- FeatureStore: Central container for feature assets
- Entity: Business objects with join keys
- FeatureView: Feature computation logic with versioning
- Point-in-time correct training data generation
- Automated lineage and governance

Example Usage:

    >>> from snowflake_ml_template.feature_store import FeatureStore, Entity, FeatureView

    >>> # Initialize feature store
    >>> fs = FeatureStore(session=session, database="ML_PROD_DB", schema="FEATURES")

    >>> # Define entity
    >>> customer_entity = Entity(
    ...     name="CUSTOMER",
    ...     join_keys=["CUSTOMER_ID"],
    ...     description="Customer entity for ML features"
    ... )
    >>> fs.register_entity(customer_entity)

    >>> # Create feature view
    >>> source_df = session.table("RAW_TRANSACTIONS")
    >>> feature_df = source_df.group_by("CUSTOMER_ID").agg(
    ...     count("*").alias("transaction_count"),
    ...     sum("AMOUNT").alias("total_amount")
    ... )

    >>> customer_features = FeatureView(
    ...     name="CUSTOMER_TRANSACTION_FEATURES",
    ...     entities=[customer_entity],
    ...     feature_df=feature_df,
    ...     refresh_freq="1 day",
    ...     description="Customer transaction summary features"
    ... )
    >>> fs.register_feature_view(customer_features)

    >>> # Generate training data with point-in-time correctness
    >>> spine_df = session.table("CUSTOMER_LABELS")
    >>> training_data = fs.generate_dataset(
    ...     spine_df=spine_df,
    ...     feature_views=["CUSTOMER_TRANSACTION_FEATURES"],
    ...     spine_timestamp_col="EVENT_DATE"
    ... )
"""

from snowflake_ml_template.feature_store.core import SQLEntity, SQLFeatureView
from snowflake_ml_template.feature_store.core.entity import Entity
from snowflake_ml_template.feature_store.core.feature_view import FeatureView
from snowflake_ml_template.feature_store.core.store import FeatureStore

__all__ = [
    "Entity",
    "FeatureStore",
    "FeatureView",
    "SQLEntity",
    "SQLFeatureView",
]

# Removed legacy imports to fix mypy errors
# The feature store now uses the core implementations directly
