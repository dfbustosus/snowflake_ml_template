"""Schema provisioning for Snowflake MLOps.

This module provides functionality to create and manage Snowflake schemas
following the canonical schema structure for ML workloads.

Classes:
    SchemaProvisioner: Create and manage schemas
"""

from typing import Dict, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import ConfigurationError
from snowflake_ml_template.utils.logging import get_logger


class SchemaProvisioner:
    """Provision and manage Snowflake schemas.

    This class handles the creation and management of Snowflake schemas
    following the Golden Migration Plan's canonical schema structure:
    - RAW_DATA: Immutable raw data from sources
    - FEATURES: Feature store tables
    - MODELS: Model registry tables
    - PIPELINES: Pipeline control and metadata
    - ANALYTICS: Analytics and reporting tables

    Attributes:
        session: Snowflake session
        logger: Structured logger

    Example:
        >>> provisioner = SchemaProvisioner(session)
        >>>
        >>> # Create canonical schemas in a database
        >>> provisioner.create_canonical_schemas("ML_DEV_DB")
        >>>
        >>> # Create a custom schema
        >>> provisioner.create_schema(
        ...     database="ML_DEV_DB",
        ...     schema="EXPERIMENTS",
        ...     comment="Experimental features"
        ... )
    """

    # Canonical schema names
    CANONICAL_SCHEMAS = ["RAW_DATA", "FEATURES", "MODELS", "PIPELINES", "ANALYTICS"]

    def __init__(self, session: Session) -> None:
        """Initialize the schema provisioner.

        Args:
            session: Active Snowflake session

        Raises:
            ValueError: If session is None
        """
        if session is None:
            raise ValueError("Session cannot be None")

        self.session = session
        self.logger = get_logger(__name__)

    def create_canonical_schemas(
        self, database: str, schemas: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Create the canonical schema structure in a database.

        This method creates the standard schema structure for ML workloads.
        By default, it creates all canonical schemas, but you can specify
        a subset if needed.

        Args:
            database: Database name
            schemas: Optional list of schema names (defaults to all canonical)

        Returns:
            Dictionary mapping schema names to creation status

        Example:
            >>> results = provisioner.create_canonical_schemas("ML_DEV_DB")
            >>> # {'RAW_DATA': True, 'FEATURES': True, ...}
        """
        if not database:
            raise ValueError("Database name cannot be empty")

        schemas_to_create = schemas or self.CANONICAL_SCHEMAS

        self.logger.info(
            f"Creating canonical schemas in database: {database}",
            extra={"schemas": schemas_to_create},
        )

        results = {}

        for schema in schemas_to_create:
            results[schema] = self.create_schema(
                database=database,
                schema=schema,
                comment=self._get_schema_comment(schema),
            )

        self.logger.info(
            f"Canonical schemas created in {database}", extra={"results": results}
        )

        return results

    def create_schema(
        self,
        database: str,
        schema: str,
        comment: Optional[str] = None,
        transient: bool = False,
        managed_access: bool = False,
    ) -> bool:
        """Create a schema if it doesn't exist.

        This method is idempotent - it will not fail if the schema
        already exists.

        Args:
            database: Database name
            schema: Schema name
            comment: Optional comment describing the schema
            transient: Whether to create a transient schema
            managed_access: Whether to use managed access

        Returns:
            True if schema was created or already exists

        Raises:
            ConfigurationError: If schema creation fails

        Example:
            >>> provisioner.create_schema(
            ...     database="ML_DEV_DB",
            ...     schema="EXPERIMENTS",
            ...     comment="Experimental features",
            ...     managed_access=True
            ... )
        """
        if not database or not schema:
            raise ValueError("Database and schema names cannot be empty")

        full_name = f"{database}.{schema}"
        self.logger.info(f"Creating schema: {full_name}")

        try:
            # Build CREATE SCHEMA statement
            sql_parts = ["CREATE"]

            if transient:
                sql_parts.append("TRANSIENT")

            sql_parts.append(f"SCHEMA IF NOT EXISTS {full_name}")

            if managed_access:
                sql_parts.append("WITH MANAGED ACCESS")

            if comment:
                sql_parts.append(f"COMMENT = '{comment}'")

            sql = " ".join(sql_parts)

            # Execute
            self.session.sql(sql).collect()

            self.logger.info(
                f"Schema created successfully: {full_name}",
                extra={"transient": transient, "managed_access": managed_access},
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to create schema: {full_name}", extra={"error": str(e)}
            )
            raise ConfigurationError(
                f"Failed to create schema: {full_name}",
                context={"database": database, "schema": schema},
                original_error=e,
            )

    def drop_schema(
        self, database: str, schema: str, if_exists: bool = True, cascade: bool = False
    ) -> bool:
        """Drop a schema.

        Args:
            database: Database name
            schema: Schema name
            if_exists: Whether to use IF EXISTS clause
            cascade: Whether to cascade drop to objects

        Returns:
            True if schema was dropped

        Raises:
            ConfigurationError: If drop fails
        """
        if not database or not schema:
            raise ValueError("Database and schema names cannot be empty")

        full_name = f"{database}.{schema}"
        self.logger.warning(f"Dropping schema: {full_name}")

        try:
            sql = f"DROP SCHEMA {'IF EXISTS' if if_exists else ''} {full_name}"
            if cascade:
                sql += " CASCADE"

            self.session.sql(sql).collect()

            self.logger.info(f"Schema dropped: {full_name}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to drop schema: {full_name}", extra={"error": str(e)}
            )
            raise ConfigurationError(
                f"Failed to drop schema: {full_name}",
                context={"database": database, "schema": schema},
                original_error=e,
            )

    def schema_exists(self, database: str, schema: str) -> bool:
        """Check if a schema exists.

        Args:
            database: Database name
            schema: Schema name

        Returns:
            True if schema exists, False otherwise
        """
        try:
            result = self.session.sql(
                f"SHOW SCHEMAS LIKE '{schema}' IN DATABASE {database}"
            ).collect()
            return len(result) > 0
        except Exception:
            return False

    def list_schemas(self, database: str) -> List[str]:
        """List all schemas in a database.

        Args:
            database: Database name

        Returns:
            List of schema names
        """
        try:
            result = self.session.sql(f"SHOW SCHEMAS IN DATABASE {database}").collect()
            return [row["name"] for row in result]
        except Exception as e:
            self.logger.error(f"Failed to list schemas in {database}: {e}")
            return []

    def _get_schema_comment(self, schema: str) -> str:
        """Get default comment for a canonical schema.

        Args:
            schema: Schema name

        Returns:
            Default comment for the schema
        """
        comments = {
            "RAW_DATA": "Immutable raw data from source systems",
            "FEATURES": "Feature store tables and feature views",
            "MODELS": "Model registry and model metadata",
            "PIPELINES": "Pipeline control tables and metadata",
            "ANALYTICS": "Analytics and reporting tables",
        }
        return comments.get(schema, f"Schema: {schema}")
