"""Configuration validator for validating Snowflake resources.

This module provides validation capabilities to ensure that configuration
references valid Snowflake resources (databases, schemas, warehouses, etc.)
and that credentials are correct.

Classes:
    ConfigValidator: Validate configuration against Snowflake resources
"""

from typing import Any, List

from snowflake.snowpark import Session

from snowflake_ml_template.core.config.models import (
    FeatureStoreConfig,
    ModelRegistryConfig,
    SnowflakeConfig,
)


class ConfigValidator:
    """Validate configuration against Snowflake resources.

    This class provides methods to validate that configuration references
    valid Snowflake resources and that credentials are correct. It can:
    - Validate Snowflake connection credentials
    - Validate database existence
    - Validate schema existence
    - Validate warehouse existence
    - Validate table existence
    - Validate stage existence

    Attributes:
        session: Snowflake session for validation
        _logger: Logger instance

    Example:
        >>> validator = ConfigValidator(session)
        >>>
        >>> # Validate Snowflake configuration
        >>> is_valid = validator.validate_snowflake_config(snowflake_config)
        >>>
        >>> # Validate specific resources
        >>> validator.validate_database("ML_DEV_DB")
        >>> validator.validate_warehouse("ML_TRAINING_WH")
    """

    def __init__(self, session: Session) -> None:
        """Initialize the configuration validator.

        Args:
            session: Active Snowflake session

        Raises:
            ValueError: If session is None
        """
        if session is None:
            raise ValueError("Session cannot be None")

        self.session = session
        self._logger = self._get_logger()

    def _get_logger(self) -> Any:
        """Get logger instance.

        This is a placeholder that will be replaced with proper
        structured logging in Day 3.

        Returns:
            Logger instance
        """
        import logging

        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def validate_snowflake_config(
        self, config: SnowflakeConfig, check_resources: bool = True
    ) -> bool:
        """Validate Snowflake configuration.

        This method validates that the Snowflake configuration is correct
        and optionally checks that referenced resources exist.

        Args:
            config: Snowflake configuration to validate
            check_resources: Whether to check resource existence

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        self._logger.info("Validating Snowflake configuration")

        # Configuration is already validated by Pydantic
        # Here we check resource existence if requested

        if check_resources:
            # Validate warehouse
            if config.warehouse:
                self.validate_warehouse(config.warehouse)

            # Validate database
            if config.database:
                self.validate_database(config.database)

            # Validate schema
            if config.database and config.schema_:
                self.validate_schema(config.database, config.schema_)

        self._logger.info("Snowflake configuration is valid")
        return True

    def validate_feature_store_config(self, config: FeatureStoreConfig) -> bool:
        """Validate feature store configuration.

        Args:
            config: Feature store configuration to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        self._logger.info("Validating feature store configuration")

        # Validate database exists
        self.validate_database(config.database)

        # Validate schema exists or can be created
        try:
            self.validate_schema(config.database, config.schema_)
        except ValueError:
            self._logger.warning(
                f"Schema {config.database}.{config.schema_} does not exist. "
                "It will be created during infrastructure setup."
            )

        self._logger.info("Feature store configuration is valid")
        return True

    def validate_model_registry_config(self, config: ModelRegistryConfig) -> bool:
        """Validate model registry configuration.

        Args:
            config: Model registry configuration to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        self._logger.info("Validating model registry configuration")

        # Validate database exists
        self.validate_database(config.database)

        # Validate schema exists or can be created
        try:
            self.validate_schema(config.database, config.schema_)
        except ValueError:
            self._logger.warning(
                f"Schema {config.database}.{config.schema_} does not exist. "
                "It will be created during infrastructure setup."
            )

        self._logger.info("Model registry configuration is valid")
        return True

    def validate_database(self, database: str) -> bool:
        """Validate that a database exists.

        Args:
            database: Database name

        Returns:
            True if database exists

        Raises:
            ValueError: If database doesn't exist
        """
        self._logger.debug(f"Validating database: {database}")

        try:
            result = self.session.sql(f"SHOW DATABASES LIKE '{database}'").collect()

            if not result:
                raise ValueError(f"Database does not exist: {database}")

            self._logger.debug(f"Database exists: {database}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to validate database {database}: {e}")
            raise ValueError(f"Database validation failed: {database}") from e

    def validate_schema(self, database: str, schema: str) -> bool:
        """Validate that a schema exists.

        Args:
            database: Database name
            schema: Schema name

        Returns:
            True if schema exists

        Raises:
            ValueError: If schema doesn't exist
        """
        self._logger.debug(f"Validating schema: {database}.{schema}")

        try:
            result = self.session.sql(
                f"SHOW SCHEMAS LIKE '{schema}' IN DATABASE {database}"
            ).collect()

            if not result:
                raise ValueError(f"Schema does not exist: {database}.{schema}")

            self._logger.debug(f"Schema exists: {database}.{schema}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to validate schema {database}.{schema}: {e}")
            raise ValueError(f"Schema validation failed: {database}.{schema}") from e

    def validate_warehouse(self, warehouse: str) -> bool:
        """Validate that a warehouse exists.

        Args:
            warehouse: Warehouse name

        Returns:
            True if warehouse exists

        Raises:
            ValueError: If warehouse doesn't exist
        """
        self._logger.debug(f"Validating warehouse: {warehouse}")

        try:
            result = self.session.sql(f"SHOW WAREHOUSES LIKE '{warehouse}'").collect()

            if not result:
                raise ValueError(f"Warehouse does not exist: {warehouse}")

            self._logger.debug(f"Warehouse exists: {warehouse}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to validate warehouse {warehouse}: {e}")
            raise ValueError(f"Warehouse validation failed: {warehouse}") from e

    def validate_table(self, database: str, schema: str, table: str) -> bool:
        """Validate that a table exists.

        Args:
            database: Database name
            schema: Schema name
            table: Table name

        Returns:
            True if table exists

        Raises:
            ValueError: If table doesn't exist
        """
        full_name = f"{database}.{schema}.{table}"
        self._logger.debug(f"Validating table: {full_name}")

        try:
            result = self.session.sql(
                f"SHOW TABLES LIKE '{table}' IN SCHEMA {database}.{schema}"
            ).collect()

            if not result:
                raise ValueError(f"Table does not exist: {full_name}")

            self._logger.debug(f"Table exists: {full_name}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to validate table {full_name}: {e}")
            raise ValueError(f"Table validation failed: {full_name}") from e

    def validate_stage(self, stage: str) -> bool:
        """Validate that a stage exists.

        Args:
            stage: Stage name (with @ prefix)

        Returns:
            True if stage exists

        Raises:
            ValueError: If stage doesn't exist
        """
        # Remove @ prefix if present
        stage_name = stage.lstrip("@")

        self._logger.debug(f"Validating stage: {stage_name}")

        try:
            result = self.session.sql(f"SHOW STAGES LIKE '{stage_name}'").collect()

            if not result:
                raise ValueError(f"Stage does not exist: {stage}")

            self._logger.debug(f"Stage exists: {stage}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to validate stage {stage}: {e}")
            raise ValueError(f"Stage validation failed: {stage}") from e

    def get_database_list(self) -> List[str]:
        """Get list of all databases.

        Returns:
            List of database names
        """
        result = self.session.sql("SHOW DATABASES").collect()
        return [row["name"] for row in result]

    def get_schema_list(self, database: str) -> List[str]:
        """Get list of all schemas in a database.

        Args:
            database: Database name

        Returns:
            List of schema names
        """
        result = self.session.sql(f"SHOW SCHEMAS IN DATABASE {database}").collect()
        return [row["name"] for row in result]

    def get_warehouse_list(self) -> List[str]:
        """Get list of all warehouses.

        Returns:
            List of warehouse names
        """
        result = self.session.sql("SHOW WAREHOUSES").collect()
        return [row["name"] for row in result]

    def get_table_list(self, database: str, schema: str) -> List[str]:
        """Get list of all tables in a schema.

        Args:
            database: Database name
            schema: Schema name

        Returns:
            List of table names
        """
        result = self.session.sql(
            f"SHOW TABLES IN SCHEMA {database}.{schema}"
        ).collect()
        return [row["name"] for row in result]
