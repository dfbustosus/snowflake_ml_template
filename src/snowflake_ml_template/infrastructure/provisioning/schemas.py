"""Schema provisioning for Snowflake MLOps.

This module provides functionality to create and manage Snowflake schemas
following the canonical schema structure for ML workloads.

Classes:
    SchemaProvisioner: Create and manage schemas
"""

from typing import Any, Dict, Iterable, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.tracking import ExecutionEventTracker
from snowflake_ml_template.infrastructure.provisioning.base import BaseProvisioner


class SchemaProvisioner(BaseProvisioner):
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

    def __init__(
        self,
        session: Session,
        tracker: Optional[ExecutionEventTracker] = None,
    ) -> None:
        """Initialize `SchemaProvisioner` with a Snowflake session."""
        super().__init__(session=session, tracker=tracker)

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
        *,
        data_retention_time_in_days: Optional[int] = None,
        default_ddl_collation: Optional[str] = None,
        default_sequence_order: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        grant_future_privileges_to: Optional[Iterable[str]] = None,
        directory: Optional[bool] = None,
        encryption: Optional[str] = None,
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
            directory: Whether to enable directory
            encryption: Type of encryption to use

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
        qualified_name = self.format_qualified_identifier(database, schema)
        self.logger.info("Creating schema", extra={"schema": full_name})

        clauses = ["CREATE"]

        if transient:
            clauses.append("TRANSIENT")

        clauses.append(f"SCHEMA IF NOT EXISTS {qualified_name}")

        options: Dict[str, Any] = {}
        if managed_access:
            clauses.append("WITH MANAGED ACCESS")
        if data_retention_time_in_days is not None:
            options["DATA_RETENTION_TIME_IN_DAYS"] = data_retention_time_in_days
        if default_ddl_collation:
            options["DEFAULT_DDL_COLLATION"] = default_ddl_collation
        if default_sequence_order:
            options["DEFAULT_SEQUENCE_ORDER"] = default_sequence_order

        set_clause = self._format_set_options(options)
        if set_clause:
            clauses.append(set_clause)

        if directory is not None:
            enable_literal = str(directory).upper()
            clauses.append(f"DIRECTORY = ( ENABLE = {enable_literal} )")

        if encryption:
            clauses.append(f"ENCRYPTION = ( TYPE = {self.quote_literal(encryption)} )")

        if comment:
            clauses.append(f"COMMENT = {self.quote_literal(comment)}")

        sql = " ".join(clauses)

        self._execute_sql(
            sql,
            context={"database": database, "schema": schema},
            emit_event="schema_created",
        )

        if tags:
            self._apply_tags("SCHEMA", qualified_name, tags)

        if grant_future_privileges_to:
            database_identifier = self.quote_identifier(database)
            for role in grant_future_privileges_to:
                self._grant_future_privileges_to_role(role, database_identifier)

        return True

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
        qualified_name = self.format_qualified_identifier(database, schema)
        self.logger.warning("Dropping schema", extra={"schema": full_name})

        clause = "IF EXISTS " if if_exists else ""
        sql = f"DROP SCHEMA {clause}{qualified_name}".strip()
        if cascade:
            sql += " CASCADE"

        self._execute_sql(
            sql,
            context={"database": database, "schema": schema},
            emit_event="schema_dropped",
        )
        return True

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
                f"SHOW SCHEMAS LIKE {self.quote_literal(schema)} IN DATABASE {self.quote_identifier(database)}"
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
            result = self.session.sql(
                f"SHOW SCHEMAS IN DATABASE {self.quote_identifier(database)}"
            ).collect()
            return [row["name"] for row in result]
        except Exception as e:
            self.logger.error(
                "Failed to list schemas",
                extra={"database": database, "error": str(e)},
            )
            return []

    def _grant_future_privileges_to_role(self, role: str, database: str) -> None:
        sql = (
            f"GRANT USAGE ON FUTURE SCHEMAS IN DATABASE {database} "
            f"TO ROLE {self.quote_identifier(role)}"
        )
        self._execute_sql(sql, context={"role": role, "database": database})

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
