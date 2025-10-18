"""Database provisioning for Snowflake MLOps.

This module provides functionality to create and manage Snowflake databases
following the three-tier environment pattern (DEV, TEST, PROD).

Classes:
    DatabaseProvisioner: Create and manage databases
"""

from typing import Dict, Iterable, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.tracking import ExecutionEventTracker
from snowflake_ml_template.infrastructure.provisioning.base import BaseProvisioner


class DatabaseProvisioner(BaseProvisioner):
    """Provision and manage Snowflake databases.

    This class handles the creation and management of Snowflake databases
    following the Golden Migration Plan's three-tier environment pattern:
    - ML_DEV_DB: Development environment
    - ML_TEST_DB: Testing environment (can be cloned from PROD)
    - ML_PROD_DB: Production environment

    All operations are idempotent - running them multiple times produces
    the same result.
    """

    def __init__(
        self,
        session: Session,
        tracker: Optional[ExecutionEventTracker] = None,
    ) -> None:
        """Initialize the provisioner with a Snowflake session."""
        super().__init__(session=session, tracker=tracker)

    def create_environment_databases(
        self,
        dev_db: str = "ML_DEV_DB",
        test_db: str = "ML_TEST_DB",
        prod_db: str = "ML_PROD_DB",
        clone_test_from_prod: bool = False,
        *,
        tags: Optional[Dict[str, str]] = None,
        data_retention_time_in_days: int = 1,
        max_data_extension_time_in_days: Optional[int] = None,
        default_ddl_collation: Optional[str] = None,
        replication_targets: Optional[List[str]] = None,
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
            name=dev_db,
            comment="Development environment for ML workloads",
            data_retention_time_in_days=data_retention_time_in_days,
            max_data_extension_time_in_days=max_data_extension_time_in_days,
            default_ddl_collation=default_ddl_collation,
            tags=tags,
            replication_targets=replication_targets,
        )

        # Create PROD database
        results[prod_db] = self.create_database(
            name=prod_db,
            comment="Production environment for ML workloads",
            data_retention_time_in_days=data_retention_time_in_days,
            max_data_extension_time_in_days=max_data_extension_time_in_days,
            default_ddl_collation=default_ddl_collation,
            tags=tags,
            replication_targets=replication_targets,
        )

        # Create TEST database (clone or new)
        if clone_test_from_prod:
            results[test_db] = self.clone_database(
                source=prod_db,
                target=test_db,
                comment="Test environment (cloned from production)",
                tags=tags,
                replication_targets=replication_targets,
            )
        else:
            results[test_db] = self.create_database(
                name=test_db,
                comment="Test environment for ML workloads",
                data_retention_time_in_days=data_retention_time_in_days,
                max_data_extension_time_in_days=max_data_extension_time_in_days,
                default_ddl_collation=default_ddl_collation,
                tags=tags,
                replication_targets=replication_targets,
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
        *,
        max_data_extension_time_in_days: Optional[int] = None,
        default_ddl_collation: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        replication_targets: Optional[List[str]] = None,
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

        database_identifier = self.quote_identifier(name)
        self.logger.info("Creating database", extra={"database": name})

        clauses = ["CREATE"]
        if transient:
            clauses.append("TRANSIENT")
        clauses.append(f"DATABASE IF NOT EXISTS {database_identifier}")

        options = {
            "DATA_RETENTION_TIME_IN_DAYS": data_retention_time_in_days,
            "MAX_DATA_EXTENSION_TIME_IN_DAYS": max_data_extension_time_in_days,
            "DEFAULT_DDL_COLLATION": default_ddl_collation,
        }
        set_clause = self._format_set_options(options)
        if set_clause:
            clauses.append(set_clause)
        if comment:
            clauses.append(f"COMMENT = {self.quote_literal(comment)}")

        sql = " ".join(clauses)

        self._execute_sql(
            sql,
            context={"database": name, "transient": transient},
            emit_event="database_created",
        )

        if tags:
            self._apply_tags("DATABASE", database_identifier, tags)

        if replication_targets:
            self._enable_replication(database_identifier, replication_targets)

        return True

    def clone_database(
        self,
        source: str,
        target: str,
        comment: Optional[str] = None,
        *,
        tags: Optional[Dict[str, str]] = None,
        replication_targets: Optional[List[str]] = None,
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

        self.logger.info("Cloning database", extra={"source": source, "target": target})

        source_identifier = self.quote_identifier(source)
        target_identifier = self.quote_identifier(target)

        sql = (
            f"CREATE OR REPLACE DATABASE {target_identifier} CLONE {source_identifier}"
        )
        if comment:
            sql += f" COMMENT = {self.quote_literal(comment)}"

        with self.transactional():
            self._execute_sql(
                sql,
                context={"source": source, "target": target},
                emit_event="database_cloned",
            )

            if tags:
                self._apply_tags("DATABASE", target_identifier, tags)

            if replication_targets:
                self._enable_replication(target_identifier, replication_targets)

        return True

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

        database_identifier = self.quote_identifier(name)
        self.logger.warning("Dropping database", extra={"database": name})

        clause = "IF EXISTS " if if_exists else ""
        sql = f"DROP DATABASE {clause}{database_identifier}".strip()

        self._execute_sql(
            sql,
            context={"database": name},
            emit_event="database_dropped",
        )
        return True

    def database_exists(self, name: str) -> bool:
        """Check if a database exists.

        Args:
            name: Database name

        Returns:
            True if database exists, False otherwise
        """
        try:
            literal = self.quote_literal(name)
            result = self.session.sql(f"SHOW DATABASES LIKE {literal}").collect()
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
            self.logger.error("Failed to list databases", extra={"error": str(e)})
            return []

    def _enable_replication(
        self, database_identifier: str, targets: Iterable[str]
    ) -> None:
        formatted_targets = []
        for target in targets:
            literal = self.quote_literal(target)
            if literal is None:
                continue
            formatted_targets.append(literal)
        if not formatted_targets:
            return
        target_clause = ", ".join(formatted_targets)
        sql = (
            f"ALTER DATABASE {database_identifier} "
            f"ENABLE REPLICATION TO ACCOUNTS {target_clause}"
        )
        self._execute_sql(sql, context={"database": database_identifier})
