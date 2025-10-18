"""Stage provisioning for Snowflake MLOps.

This module provides functionality to create and manage Snowflake stages
for storing model artifacts and data files.

Classes:
    StageProvisioner: Create and manage stages
"""

from typing import Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import ConfigurationError
from snowflake_ml_template.utils.logging import get_logger


class StageProvisioner:
    """Provision and manage Snowflake stages."""

    def __init__(self, session: Session) -> None:
        """Initialize the StageProvisioner with a Snowflake session.

        Args:
            session: An active Snowflake session for executing stage operations.

        Raises:
            ValueError: If the provided session is None.
        """
        if session is None:
            raise ValueError("Session cannot be None")
        self.session = session
        self.logger = get_logger(__name__)

    def create_stage(
        self,
        name: str,
        database: str,
        schema: str,
        url: Optional[str] = None,
        storage_integration: Optional[str] = None,
        file_format: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> bool:
        """Create a stage if it doesn't exist."""
        if not name or not database or not schema:
            raise ValueError("Name, database, and schema cannot be empty")

        full_name = f"{database}.{schema}.{name}"
        self.logger.info(f"Creating stage: {full_name}")

        try:
            sql_parts = [f"CREATE STAGE IF NOT EXISTS {full_name}"]

            if url:
                sql_parts.append(f"URL = '{url}'")
                if storage_integration:
                    sql_parts.append(f"STORAGE_INTEGRATION = {storage_integration}")

            if file_format:
                sql_parts.append(f"FILE_FORMAT = {file_format}")

            if comment:
                sql_parts.append(f"COMMENT = '{comment}'")

            sql = " ".join(sql_parts)
            self.session.sql(sql).collect()

            self.logger.info(f"Stage created: {full_name}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to create stage: {full_name}", extra={"error": str(e)}
            )
            raise ConfigurationError(
                f"Failed to create stage: {full_name}",
                context={"stage": full_name},
                original_error=e,
            )

    def stage_exists(self, name: str, database: str, schema: str) -> bool:
        """Check if a stage exists."""
        try:
            result = self.session.sql(
                f"SHOW STAGES LIKE '{name}' IN SCHEMA {database}.{schema}"
            ).collect()
            return len(result) > 0
        except Exception:
            return False
