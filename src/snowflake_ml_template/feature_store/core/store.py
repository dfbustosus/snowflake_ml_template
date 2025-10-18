"""Core FeatureStore implementation."""

from typing import Dict, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import FeatureStoreError
from snowflake_ml_template.feature_store.core.entity import Entity
from snowflake_ml_template.feature_store.core.feature_view import FeatureView
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """Enterprise Feature Store for Snowflake ML."""

    def __init__(self, session: Session, database: str, schema: str = "FEATURES"):
        """Initialize the FeatureStore."""
        if session is None:
            raise ValueError("Session cannot be None")
        if not database:
            raise ValueError("Database cannot be empty")

        self.session = session
        self.database = database
        self.schema = schema
        self._entities: Dict[str, Entity] = {}
        self._feature_views: Dict[str, FeatureView] = {}

        self._ensure_schema_exists()
        logger.info(f"Initialized Feature Store: {self.database}.{self.schema}")

    def _ensure_schema_exists(self) -> None:
        """Ensure the feature store schema exists."""
        try:
            self.session.sql(
                f"CREATE SCHEMA IF NOT EXISTS {self.database}.{self.schema}"
            ).collect()
            self.session.sql(f"USE SCHEMA {self.database}.{self.schema}").collect()
        except Exception as e:
            raise FeatureStoreError(
                f"Failed to create schema {self.database}.{self.schema}",
                original_error=e,
            )

    def register_entity(self, entity: Entity) -> None:
        """Register an entity in the feature store."""
        if entity.name in self._entities:
            logger.warning(f"Entity {entity.name} already registered, updating")

        self._entities[entity.name] = entity
        self._create_entity_metadata_table(entity)
        logger.info(
            f"Registered entity: {entity.name} with join keys: {entity.join_keys}"
        )

    def _create_entity_metadata_table(self, entity: Entity) -> None:
        """Create metadata table for entity."""
        table_name = f"ENTITY_{entity.name}_METADATA"
        metadata_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.{table_name} (
            entity_name VARCHAR,
            join_keys ARRAY,
            description VARCHAR,
            owner VARCHAR,
            tags VARIANT,
            created_at TIMESTAMP_NTZ,
            updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        self.session.sql(metadata_sql).collect()

    def register_feature_view(
        self, feature_view: FeatureView, overwrite: bool = False
    ) -> None:
        """Register a feature view in the feature store."""
        if feature_view.name in self._feature_views and not overwrite:
            raise FeatureStoreError(
                f"FeatureView {feature_view.name} already exists. Use overwrite=True to replace."
            )

        self._feature_views[feature_view.name] = feature_view

        if feature_view.is_snowflake_managed:
            self._create_dynamic_table(feature_view, overwrite)
        else:
            self._validate_external_table(feature_view)

        self._create_feature_view_metadata(feature_view)
        logger.info(
            f"Registered FeatureView: {feature_view.name} ({'Snowflake-managed' if feature_view.is_snowflake_managed else 'External'})"
        )

    def _create_dynamic_table(self, feature_view: FeatureView, overwrite: bool) -> None:
        """Create a Snowflake-managed Dynamic Table."""
        table_name = f"FEATURE_VIEW_{feature_view.name}_V{feature_view.version.replace('.', '_')}"

        if overwrite:
            self.session.sql(
                f"DROP TABLE IF EXISTS {self.database}.{self.schema}.{table_name}"
            ).collect()

        feature_view.feature_df.write.mode("overwrite").save_as_table(
            f"{self.database}.{self.schema}.{table_name}"
        )
        logger.info(f"Created table: {table_name}")

    def _validate_external_table(self, feature_view: FeatureView) -> None:
        """Validate that external table exists."""
        table_name = f"FEATURE_VIEW_{feature_view.name}_V{feature_view.version.replace('.', '_')}"
        try:
            result = self.session.sql(
                f"SHOW TABLES LIKE '{table_name}' IN SCHEMA {self.database}.{self.schema}"
            ).collect()
            if not result:
                raise FeatureStoreError(f"External table {table_name} does not exist")
        except Exception as e:
            raise FeatureStoreError(
                f"Error validating external table {table_name}", original_error=e
            )

    def _create_feature_view_metadata(self, feature_view: FeatureView) -> None:
        """Create metadata table for feature view."""
        table_name = f"FEATURE_VIEW_{feature_view.name}_METADATA"
        metadata_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.{table_name} (
            feature_view_name VARCHAR,
            version VARCHAR,
            entities ARRAY,
            feature_names ARRAY,
            refresh_freq VARCHAR,
            description VARCHAR,
            owner VARCHAR,
            tags VARIANT,
            created_at TIMESTAMP_NTZ,
            updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        self.session.sql(metadata_sql).collect()

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        return self._entities.get(name)

    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        """Get a feature view by name."""
        return self._feature_views.get(name)

    def list_entities(self) -> List[str]:
        """List all registered entities."""
        return list(self._entities.keys())

    def list_feature_views(self) -> List[str]:
        """List all registered feature views."""
        return list(self._feature_views.keys())
