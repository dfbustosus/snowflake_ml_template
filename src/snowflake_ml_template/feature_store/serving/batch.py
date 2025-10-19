"""Batch feature serving with point-in-time correctness using native ASOF JOIN."""

from typing import List, Optional, Tuple, cast

from snowflake.snowpark import DataFrame, Session

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
        *,
        asof_tolerance: Optional[str] = None,
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

        spine_sql = self._resolve_dataframe_sql(spine_df)
        base_columns = list(getattr(spine_df, "columns", []))
        current_sql = spine_sql
        result_columns = base_columns[:]

        for idx, fv_name in enumerate(feature_views):
            fv_sql, fv_columns = self._get_feature_view_sql_and_columns(fv_name)
            entity_columns = self._determine_entity_columns(
                spine_entity_cols, fv_columns, base_columns, spine_timestamp_col
            )

            feature_columns = [
                col_name
                for col_name in fv_columns
                if col_name not in entity_columns
                and (spine_timestamp_col is None or col_name != spine_timestamp_col)
            ]

            if not feature_columns:
                continue

            if spine_timestamp_col:
                current_sql = self._build_asof_join_query(
                    left_sql=current_sql,
                    right_sql=fv_sql,
                    left_alias=f"SPINE_{idx}",
                    right_alias=f"FV_{idx}",
                    entity_cols=entity_columns,
                    timestamp_col=spine_timestamp_col,
                    feature_columns=feature_columns,
                    tolerance=asof_tolerance,
                )
            else:
                current_sql = self._build_left_join_query(
                    left_sql=current_sql,
                    right_sql=fv_sql,
                    left_alias=f"SPINE_{idx}",
                    right_alias=f"FV_{idx}",
                    entity_cols=entity_columns,
                    feature_columns=feature_columns,
                )

            for col_name in feature_columns:
                if col_name not in result_columns:
                    result_columns.append(col_name)

        result_df = self._execute_sql(current_sql)

        # Populate columns for stubs used in unit tests
        if hasattr(result_df, "set_columns"):
            set_columns = getattr(result_df, "set_columns")
            set_columns(result_columns)
        elif hasattr(result_df, "columns") and not getattr(result_df, "columns"):
            try:
                setattr(result_df, "columns", result_columns)
            except Exception as exc:  # pragma: no cover - safety
                self.logger.warning(
                    "Failed to set columns on result DataFrame stubs",
                    extra={"error": str(exc)},
                )

        self.logger.info(
            "Retrieved features",
            extra={
                "feature_views": feature_views,
                "spine_timestamp_col": spine_timestamp_col,
                "asof_tolerance": asof_tolerance,
            },
        )

        return result_df

    def generate_training_dataset(
        self,
        spine_df: DataFrame,
        feature_views: List[str],
        label_col: str,
        spine_timestamp_col: str,
        exclude_cols: Optional[List[str]] = None,
        *,
        spine_entity_cols: Optional[List[str]] = None,
        asof_tolerance: Optional[str] = None,
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
            spine_entity_cols=spine_entity_cols,
            asof_tolerance=asof_tolerance,
        )

        # Select columns
        exclude_cols = exclude_cols or []
        cols_to_keep = [
            c for c in dataset.columns if c not in exclude_cols or c == label_col
        ]

        result = cast(DataFrame, dataset.select(cols_to_keep))

        self.logger.info(
            f"Generated training dataset with {len(feature_views)} feature views",
            extra={
                "feature_views": feature_views,
                "label_col": label_col,
                "num_columns": len(cols_to_keep),
            },
        )

        return result

    def generate_inference_dataset(
        self,
        spine_df: DataFrame,
        feature_views: List[str],
        spine_timestamp_col: str,
        spine_entity_cols: Optional[List[str]] = None,
        *,
        asof_tolerance: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
    ) -> DataFrame:
        """Generate an inference-ready dataset with point-in-time features only."""
        dataset = self.get_features(
            spine_df=spine_df,
            feature_views=feature_views,
            spine_timestamp_col=spine_timestamp_col,
            spine_entity_cols=spine_entity_cols,
            asof_tolerance=asof_tolerance,
        )

        exclude_cols = exclude_cols or []
        entity_cols = spine_entity_cols or []
        cols_to_drop = set(exclude_cols + entity_cols + [spine_timestamp_col])
        projection = [
            col_name for col_name in dataset.columns if col_name not in cols_to_drop
        ]

        result = cast(DataFrame, dataset.select(projection))

        self.logger.info(
            "Generated inference dataset",
            extra={
                "feature_views": feature_views,
                "num_columns": len(projection),
            },
        )

        return result

    # ------------------------------------------------------------------
    # SQL builders & helpers
    # ------------------------------------------------------------------
    def _resolve_dataframe_sql(self, dataframe: DataFrame) -> str:
        if hasattr(dataframe, "to_sql"):
            sql = dataframe.to_sql()
            if isinstance(sql, tuple):
                sql = sql[0]
            if isinstance(sql, str):
                return sql.strip().rstrip(";")

        plan = getattr(dataframe, "_plan", None)
        if plan is not None:
            plan_queries = getattr(plan, "queries", None)
            if isinstance(plan_queries, list) and plan_queries:
                sql = plan_queries[-1]
                if isinstance(sql, str):
                    return sql.strip().rstrip(";")
                if isinstance(sql, dict):
                    text = sql.get("query") or sql.get("text")
                    if text:
                        return str(text).strip().rstrip(";")

        if hasattr(dataframe, "queries"):
            queries = getattr(dataframe, "queries")
            if isinstance(queries, list) and queries:
                sql = queries[-1]
                if isinstance(sql, str):
                    return sql.strip().rstrip(";")

        # Fallback to table reference if available
        if hasattr(dataframe, "_name"):
            return f"SELECT * FROM {dataframe._name}"

        raise FeatureStoreError(
            "Unable to derive SQL for DataFrame. Provide a Snowpark DataFrame with 'to_sql()' support."
        )

    def _get_feature_view_sql_and_columns(
        self, feature_view_name: str
    ) -> Tuple[str, List[str]]:
        """Get feature view SQL and columns."""
        table_name = f"{self.database}.{self.schema}.FEATURE_VIEW_{feature_view_name}"
        try:
            fv_df = self.session.table(table_name)
        except Exception as exc:
            raise FeatureStoreError(
                f"Feature view table not found: {table_name}", original_error=exc
            )

        fv_columns = list(getattr(fv_df, "columns", []))
        return f"SELECT * FROM {table_name}", fv_columns

    def _determine_entity_columns(
        self,
        explicit_entity_cols: Optional[List[str]],
        feature_columns: List[str],
        spine_columns: List[str],
        spine_timestamp_col: Optional[str],
    ) -> List[str]:
        if explicit_entity_cols:
            return explicit_entity_cols

        inferred = [
            column
            for column in feature_columns
            if column in spine_columns and column != spine_timestamp_col
        ]

        if not inferred:
            raise FeatureStoreError(
                "Unable to determine entity join columns. Provide 'spine_entity_cols'."
            )

        return inferred

    def _build_asof_join_query(
        self,
        *,
        left_sql: str,
        right_sql: str,
        left_alias: str,
        right_alias: str,
        entity_cols: List[str],
        timestamp_col: str,
        feature_columns: List[str],
        tolerance: Optional[str],
    ) -> str:
        select_clause = ", ".join(
            [f"{left_alias}.*"]
            + [
                f"{right_alias}.{self._quote_identifier(col)} AS {self._quote_identifier(col)}"
                for col in feature_columns
            ]
        )

        on_clause = " AND ".join(
            f"{left_alias}.{self._quote_identifier(col)} = {right_alias}.{self._quote_identifier(col)}"
            for col in entity_cols
        )

        match_conditions = [
            f"{right_alias}.{self._quote_identifier(timestamp_col)} <= {left_alias}.{self._quote_identifier(timestamp_col)}"
        ]
        if tolerance:
            match_conditions.append(
                f"{left_alias}.{self._quote_identifier(timestamp_col)} - {right_alias}.{self._quote_identifier(timestamp_col)} <= {tolerance}"
            )
        match_clause = " AND ".join(match_conditions)

        return (
            f"SELECT {select_clause} "
            f"FROM ({left_sql}) AS {left_alias} "
            f"LEFT ASOF JOIN ({right_sql}) AS {right_alias} "
            f"MATCH_CONDITION => ({match_clause}) "
            f"ON {on_clause}"
        )

    def _build_left_join_query(
        self,
        *,
        left_sql: str,
        right_sql: str,
        left_alias: str,
        right_alias: str,
        entity_cols: List[str],
        feature_columns: List[str],
    ) -> str:
        select_clause = ", ".join(
            [f"{left_alias}.*"]
            + [
                f"{right_alias}.{self._quote_identifier(col)} AS {self._quote_identifier(col)}"
                for col in feature_columns
            ]
        )

        on_clause = " AND ".join(
            f"{left_alias}.{self._quote_identifier(col)} = {right_alias}.{self._quote_identifier(col)}"
            for col in entity_cols
        )

        return (
            f"SELECT {select_clause} "
            f"FROM ({left_sql}) AS {left_alias} "
            f"LEFT JOIN ({right_sql}) AS {right_alias} "
            f"ON {on_clause}"
        )

    def _execute_sql(self, sql: str) -> DataFrame:
        self.logger.debug("Executing SQL", extra={"sql": sql})
        return self.session.sql(sql)

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'
