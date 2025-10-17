"""Batch feature serving with point-in-time correctness.

This module implements batch feature serving using ASOF joins to ensure
point-in-time correctness, preventing data leakage in training datasets.

The key principle: Features are joined based on the timestamp when they
were available, not when they were created. This ensures that training
data only uses information that would have been available at prediction time.
"""

from typing import List, Optional

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark.functions import col

from snowflake_ml_template.core.exceptions import FeatureStoreError
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class BatchFeatureServer:
    """Serve features for batch training and inference.

    This class provides point-in-time correct feature serving using ASOF
    joins. It ensures that training datasets only include features that
    would have been available at the time of prediction, preventing
    data leakage.

    Key concepts:
    - Spine DataFrame: Contains entity IDs and timestamps for which to retrieve features
    - ASOF Join: Joins features based on the latest available value at spine timestamp
    - Point-in-time correctness: No future information leakage

    Attributes:
        session: Snowflake session
        database: Database containing feature store
        schema: Schema containing feature store
        logger: Structured logger

    Example:
        >>> server = BatchFeatureServer(session, "ML_PROD_DB", "FEATURES")
        >>>
        >>> # Create spine with entity IDs and timestamps
        >>> spine_df = session.table("TRAINING_LABELS").select(
        ...     col("CUSTOMER_ID"),
        ...     col("EVENT_TIMESTAMP"),
        ...     col("LABEL")
        ... )
        >>>
        >>> # Get features with point-in-time correctness
        >>> training_data = server.get_features(
        ...     spine_df=spine_df,
        ...     feature_views=["customer_features_v1", "transaction_features_v1"],
        ...     spine_timestamp_col="EVENT_TIMESTAMP"
        ... )
    """

    def __init__(self, session: Session, database: str, schema: str):
        """Initialize the batch feature server.

        Args:
            session: Active Snowflake session
            database: Database containing feature store
            schema: Schema containing feature store
        """
        if session is None:
            raise ValueError("Session cannot be None")
        if not database or not schema:
            raise ValueError("Database and schema cannot be empty")

        self.session = session
        self.database = database
        self.schema = schema
        self.logger = get_logger(__name__)

    def get_features(
        self,
        spine_df: DataFrame,
        feature_views: List[str],
        spine_timestamp_col: Optional[str] = None,
        spine_entity_cols: Optional[List[str]] = None,
    ) -> DataFrame:
        """Get features for a spine DataFrame with point-in-time correctness.

        This method performs ASOF joins to retrieve features as they existed
        at the spine timestamp, ensuring no data leakage.

        Args:
            spine_df: Spine DataFrame with entity IDs and timestamps
            feature_views: List of feature view names to join
            spine_timestamp_col: Column in spine containing timestamps
            spine_entity_cols: Columns in spine containing entity IDs

        Returns:
            DataFrame with spine columns and features

        Raises:
            FeatureStoreError: If feature views don't exist or join fails

        Example:
            >>> # Training data generation
            >>> spine = session.table("LABELS").select(
            ...     col("CUSTOMER_ID"),
            ...     col("EVENT_DATE"),
            ...     col("CHURNED")
            ... )
            >>>
            >>> training_data = server.get_features(
            ...     spine_df=spine,
            ...     feature_views=["customer_features"],
            ...     spine_timestamp_col="EVENT_DATE"
            ... )
        """
        if not feature_views:
            raise ValueError("At least one feature view must be specified")

        result_df = spine_df

        for fv_name in feature_views:
            result_df = self._join_feature_view(
                result_df, fv_name, spine_timestamp_col, spine_entity_cols
            )

        self.logger.info(
            f"Retrieved features for {len(feature_views)} feature views",
            extra={
                "feature_views": feature_views,
                "spine_timestamp_col": spine_timestamp_col,
            },
        )

        return result_df

    def _join_feature_view(
        self,
        spine_df: DataFrame,
        feature_view_name: str,
        spine_timestamp_col: Optional[str],
        spine_entity_cols: Optional[List[str]],
    ) -> DataFrame:
        """Join a single feature view to the spine.

        Args:
            spine_df: Current spine DataFrame
            feature_view_name: Name of feature view to join
            spine_timestamp_col: Timestamp column for ASOF join
            spine_entity_cols: Entity columns for join

        Returns:
            DataFrame with feature view joined
        """
        # Get feature view table
        fv_table_name = (
            f"{self.database}.{self.schema}.FEATURE_VIEW_{feature_view_name}"
        )

        try:
            fv_df = self.session.table(fv_table_name)
        except Exception as e:
            raise FeatureStoreError(
                f"Feature view table not found: {fv_table_name}", original_error=e
            )

        # Perform join
        if spine_timestamp_col:
            # ASOF join for point-in-time correctness
            result_df = self._asof_join(
                spine_df, fv_df, spine_timestamp_col, spine_entity_cols
            )
        else:
            # Simple join (for non-temporal features)
            result_df = self._simple_join(spine_df, fv_df, spine_entity_cols)

        return result_df

    def _asof_join(
        self,
        spine_df: DataFrame,
        feature_df: DataFrame,
        timestamp_col: str,
        entity_cols: Optional[List[str]],
    ) -> DataFrame:
        """Perform ASOF join for point-in-time correctness.

        ASOF join retrieves the latest feature values that were available
        at or before the spine timestamp, ensuring no future information
        is used in training.

        Args:
            spine_df: Spine DataFrame
            feature_df: Feature DataFrame
            timestamp_col: Timestamp column name
            entity_cols: Entity column names for join

        Returns:
            DataFrame with ASOF joined features
        """
        # Use Snowflake's ASOF JOIN capability
        # This is a simplified implementation - production would use
        # Snowflake's native ASOF JOIN syntax

        # For now, use a window function approach
        from snowflake.snowpark.functions import row_number
        from snowflake.snowpark.window import Window

        # Determine entity columns
        if not entity_cols:
            # Auto-detect from feature_df (exclude timestamp and feature columns)
            entity_cols = [
                c
                for c in feature_df.columns
                if c != timestamp_col and not c.startswith("FEATURE_")
            ]

        # Join on entity columns with timestamp constraint
        join_condition = [
            spine_df[col_name] == feature_df[col_name] for col_name in entity_cols
        ]
        join_condition.append(feature_df[timestamp_col] <= spine_df[timestamp_col])

        # Perform join
        joined = spine_df.join(feature_df, join_condition, join_type="left")

        # For each spine row, keep only the latest feature value
        # (This is a simplified approach - production would use ASOF JOIN)
        window_spec = Window.partition_by(
            [spine_df[col_name] for col_name in entity_cols] + [spine_df[timestamp_col]]
        ).order_by(feature_df[timestamp_col].desc())

        # Add row number and filter to keep only the latest
        result = (
            joined.with_column("_row_num", row_number().over(window_spec))
            .filter(col("_row_num") == 1)
            .drop("_row_num")
        )

        return result

    def _simple_join(
        self,
        spine_df: DataFrame,
        feature_df: DataFrame,
        entity_cols: Optional[List[str]],
    ) -> DataFrame:
        """Perform simple join without temporal constraints.

        Used for features that don't have temporal aspects.

        Args:
            spine_df: Spine DataFrame
            feature_df: Feature DataFrame
            entity_cols: Entity column names for join

        Returns:
            DataFrame with joined features
        """
        if not entity_cols:
            # Auto-detect entity columns
            entity_cols = [
                c for c in feature_df.columns if not c.startswith("FEATURE_")
            ]

        # Simple left join on entity columns
        join_condition = [
            spine_df[col_name] == feature_df[col_name] for col_name in entity_cols
        ]

        return spine_df.join(feature_df, join_condition, join_type="left")

    def generate_training_dataset(
        self,
        spine_df: DataFrame,
        feature_views: List[str],
        label_col: str,
        spine_timestamp_col: str,
        exclude_cols: Optional[List[str]] = None,
    ) -> DataFrame:
        """Generate a complete training dataset.

        This is a convenience method that:
        1. Joins all feature views with point-in-time correctness
        2. Selects only feature columns and label
        3. Removes any specified columns

        Args:
            spine_df: Spine with entity IDs, timestamps, and labels
            feature_views: List of feature view names
            label_col: Name of the label column
            spine_timestamp_col: Timestamp column for ASOF join
            exclude_cols: Columns to exclude from final dataset

        Returns:
            Training-ready DataFrame with features and label

        Example:
            >>> training_data = server.generate_training_dataset(
            ...     spine_df=labels_df,
            ...     feature_views=["customer_features", "transaction_features"],
            ...     label_col="CHURNED",
            ...     spine_timestamp_col="EVENT_DATE",
            ...     exclude_cols=["CUSTOMER_ID", "EVENT_DATE"]
            ... )
        """
        # Get features
        dataset = self.get_features(
            spine_df=spine_df,
            feature_views=feature_views,
            spine_timestamp_col=spine_timestamp_col,
        )

        # Select columns
        exclude_cols = exclude_cols or []
        cols_to_keep = [
            c for c in dataset.columns if c not in exclude_cols or c == label_col
        ]

        result = dataset.select(cols_to_keep)

        self.logger.info(
            f"Generated training dataset with {len(feature_views)} feature views",
            extra={
                "feature_views": feature_views,
                "label_col": label_col,
                "num_columns": len(cols_to_keep),
            },
        )

        return result
