"""Database provisioning for Snowflake MLOps.

This module provides functionality to create and manage Snowflake databases
following the three-tier environment pattern (DEV, TEST, PROD).

Classes:
    DatabaseProvisioner: Create and manage databases
"""

from typing import Dict, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import ConfigurationError
from snowflake_ml_template.utils.logging import get_logger


class DatabaseProvisioner:
    """Provision and manage Snowflake databases.

    This class handles the creation and management of Snowflake databases
    following the Golden Migration Plan's three-tier environment pattern:
    - ML_DEV_DB: Development environment
    - ML_TEST_DB: Testing environment (can be cloned from PROD)
    - ML_PROD_DB: Production environment

    All operations are idempotent - running them multiple times produces
    the same result.

    Attributes:
        session: Snowflake session
        logger: Structured logger

    Example:
        >>> provisioner = DatabaseProvisioner(session)
        >>>
        >>> # Create all three environment databases
        >>> provisioner.create_environment_databases()
        >>>
        >>> # Create a custom database
        >>> provisioner.create_database(
        ...     name="CUSTOM_DB",
        ...     comment="Custom database for experiments"
        ... )
        >>>
        >>> # Clone production to test
        >>> provisioner.clone_database(
        ...     source="ML_PROD_DB",
        ...     target="ML_TEST_DB"
        ... )
    """

    def __init__(self, session: Session) -> None:
        """Initialize the database provisioner.

        Args:
            session: Active Snowflake session

        Raises:
            ValueError: If session is None
        """
        if session is None:
            raise ValueError("Session cannot be None")

        self.session = session
        self.logger = get_logger(__name__)

    def create_environment_databases(
        self,
        dev_db: str = "ML_DEV_DB",
        test_db: str = "ML_TEST_DB",
        prod_db: str = "ML_PROD_DB",
        clone_test_from_prod: bool = False,
    ) -> Dict[str, bool]:
        """Create the three-tier environment databases.

        This method creates the standard three-tier database structure
        for MLOps environments. Optionally, TEST can be created as a
        zero-copy clone of PROD.

        Args:
            dev_db: Name for development database
            test_db: Name for test database
            prod_db: Name for production database
            clone_test_from_prod: Whether to clone TEST from PROD

        Returns:
            Dictionary mapping database names to creation status

        Example:
            >>> results = provisioner.create_environment_databases()
            >>> # {'ML_DEV_DB': True, 'ML_TEST_DB': True, 'ML_PROD_DB': True}
        """
        self.logger.info(
            "Creating environment databases",
            extra={
                "dev_db": dev_db,
                "test_db": test_db,
                "prod_db": prod_db,
                "clone_test": clone_test_from_prod,
            },
        )

        results = {}

        # Create DEV database
        results[dev_db] = self.create_database(
            name=dev_db, comment="Development environment for ML workloads"
        )

        # Create PROD database
        results[prod_db] = self.create_database(
            name=prod_db, comment="Production environment for ML workloads"
        )

        # Create TEST database (clone or new)
        if clone_test_from_prod:
            results[test_db] = self.clone_database(
                source=prod_db,
                target=test_db,
                comment="Test environment (cloned from production)",
            )
        else:
            results[test_db] = self.create_database(
                name=test_db, comment="Test environment for ML workloads"
            )

        self.logger.info(
            "Environment databases created successfully", extra={"results": results}
        )

        return results

    def create_database(
        self,
        name: str,
        comment: Optional[str] = None,
        data_retention_time_in_days: int = 1,
        transient: bool = False,
    ) -> bool:
        """Create a database if it doesn't exist.

        This method is idempotent - it will not fail if the database
        already exists.

        Args:
            name: Database name
            comment: Optional comment describing the database
            data_retention_time_in_days: Time travel retention (1-90 days)
            transient: Whether to create a transient database

        Returns:
            True if database was created or already exists

        Raises:
            ConfigurationError: If database creation fails

        Example:
            >>> provisioner.create_database(
            ...     name="ANALYTICS_DB",
            ...     comment="Analytics database",
            ...     data_retention_time_in_days=7
            ... )
        """
        if not name:
            raise ValueError("Database name cannot be empty")

        self.logger.info(f"Creating database: {name}")

        try:
            # Build CREATE DATABASE statement
            sql_parts = ["CREATE"]

            if transient:
                sql_parts.append("TRANSIENT")

            sql_parts.append(f"DATABASE IF NOT EXISTS {name}")

            if data_retention_time_in_days:
                sql_parts.append(
                    f"DATA_RETENTION_TIME_IN_DAYS = {data_retention_time_in_days}"
                )

            if comment:
                sql_parts.append(f"COMMENT = '{comment}'")

            sql = " ".join(sql_parts)

            # Execute
            self.session.sql(sql).collect()

            self.logger.info(
                f"Database created successfully: {name}",
                extra={
                    "transient": transient,
                    "retention_days": data_retention_time_in_days,
                },
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to create database: {name}", extra={"error": str(e)}
            )
            raise ConfigurationError(
                f"Failed to create database: {name}",
                context={"database": name},
                original_error=e,
            )

    def clone_database(
        self, source: str, target: str, comment: Optional[str] = None
    ) -> bool:
        """Clone a database using zero-copy cloning.

        This method creates a zero-copy clone of a source database.
        This is useful for creating TEST environments from PROD.

        Args:
            source: Source database name
            target: Target database name
            comment: Optional comment for the cloned database

        Returns:
            True if database was cloned successfully

        Raises:
            ConfigurationError: If cloning fails

        Example:
            >>> provisioner.clone_database(
            ...     source="ML_PROD_DB",
            ...     target="ML_TEST_DB"
            ... )
        """
        if not source or not target:
            raise ValueError("Source and target database names cannot be empty")

        self.logger.info(f"Cloning database: {source} -> {target}")

        try:
            # Build CLONE statement
            sql = f"CREATE OR REPLACE DATABASE {target} CLONE {source}"

            if comment:
                sql += f" COMMENT = '{comment}'"

            # Execute
            self.session.sql(sql).collect()

            self.logger.info(f"Database cloned successfully: {source} -> {target}")

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to clone database: {source} -> {target}",
                extra={"error": str(e)},
            )
            raise ConfigurationError(
                f"Failed to clone database: {source} -> {target}",
                context={"source": source, "target": target},
                original_error=e,
            )

    def drop_database(self, name: str, if_exists: bool = True) -> bool:
        """Drop a database.

        Args:
            name: Database name
            if_exists: Whether to use IF EXISTS clause

        Returns:
            True if database was dropped

        Raises:
            ConfigurationError: If drop fails
        """
        if not name:
            raise ValueError("Database name cannot be empty")

        self.logger.warning(f"Dropping database: {name}")

        try:
            sql = f"DROP DATABASE {'IF EXISTS' if if_exists else ''} {name}"
            self.session.sql(sql).collect()

            self.logger.info(f"Database dropped: {name}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to drop database: {name}", extra={"error": str(e)}
            )
            raise ConfigurationError(
                f"Failed to drop database: {name}",
                context={"database": name},
                original_error=e,
            )

    def database_exists(self, name: str) -> bool:
        """Check if a database exists.

        Args:
            name: Database name

        Returns:
            True if database exists, False otherwise
        """
        try:
            result = self.session.sql(f"SHOW DATABASES LIKE '{name}'").collect()
            return len(result) > 0
        except Exception:
            return False

    def list_databases(self) -> List[str]:
        """List all databases.

        Returns:
            List of database names
        """
        try:
            result = self.session.sql("SHOW DATABASES").collect()
            return [row["name"] for row in result]
        except Exception as e:
            self.logger.error(f"Failed to list databases: {e}")
            return []
