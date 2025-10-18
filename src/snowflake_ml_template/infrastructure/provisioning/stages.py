"""Stage provisioning for Snowflake MLOps.

This module provides functionality to create and manage Snowflake stages
for storing model artifacts and data files.

Classes:
    StageProvisioner: Create and manage stages
"""

from typing import Dict, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.tracking import ExecutionEventTracker
from snowflake_ml_template.infrastructure.provisioning.base import BaseProvisioner


class StageProvisioner(BaseProvisioner):
    """Provision and manage Snowflake stages."""

    def __init__(
        self,
        session: Session,
        tracker: Optional[ExecutionEventTracker] = None,
    ) -> None:
        """Initialize `StageProvisioner` with a Snowflake session."""
        super().__init__(session=session, tracker=tracker)

    def create_stage(
        self,
        name: str,
        database: str,
        schema: str,
        url: Optional[str] = None,
        storage_integration: Optional[str] = None,
        file_format: Optional[str] = None,
        comment: Optional[str] = None,
        *,
        directory: Optional[str] = None,
        encryption: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Create a stage if it doesn't exist."""
        if not name or not database or not schema:
            raise ValueError("Name, database, and schema cannot be empty")

        qualified_name = self.format_qualified_identifier(database, schema, name)
        self.logger.info(
            "Creating stage",
            extra={"stage": qualified_name, "has_url": bool(url)},
        )

        clauses = [f"CREATE STAGE IF NOT EXISTS {qualified_name}"]

        if url:
            literal_url = self.quote_literal(url)
            clauses.append(f"URL = {literal_url}")
            if storage_integration:
                clauses.append(
                    f"STORAGE_INTEGRATION = {self.quote_identifier(storage_integration)}"
                )

        if file_format:
            clauses.append(f"FILE_FORMAT = {self.quote_identifier(file_format)}")

        if directory:
            clauses.append(f"DIRECTORY = ( ENABLE = {directory} )")

        if encryption:
            clauses.append(f"ENCRYPTION = ( TYPE = {encryption} )")

        if comment:
            clauses.append(f"COMMENT = {self.quote_literal(comment)}")

        sql = " ".join(clauses)

        rollback_statements = [f"DROP STAGE IF EXISTS {qualified_name}"]
        with self.transactional(rollback=rollback_statements):
            self._execute_sql(
                sql,
                context={"stage": qualified_name},
                emit_event="stage_created",
            )

            if tags:
                self._apply_tags("STAGE", qualified_name, tags)

        return True

    def stage_exists(self, name: str, database: str, schema: str) -> bool:
        """Check if a stage exists."""
        try:
            result = self.session.sql(
                f"SHOW STAGES LIKE {self.quote_literal(name)} "
                f"IN SCHEMA {self.format_qualified_identifier(database, schema)}"
            ).collect()
            return len(result) > 0
        except Exception:
            return False
