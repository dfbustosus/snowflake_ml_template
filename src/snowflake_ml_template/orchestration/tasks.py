"""Task orchestration using Snowflake Tasks."""

from typing import List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class TaskOrchestrator:
    """Orchestrate pipeline execution using Snowflake Tasks.

    Example:
        >>> orchestrator = TaskOrchestrator(session, "ML_PROD_DB", "PIPELINES")
        >>> orchestrator.create_task_dag(
        ...     name="fraud_detection_dag",
        ...     schedule="USING CRON 0 2 * * * UTC",
        ...     tasks=[
        ...         {"name": "ingest", "sql": "CALL ingest_data()"},
        ...         {"name": "train", "sql": "CALL train_model()", "after": ["ingest"]},
        ...         {"name": "deploy", "sql": "CALL deploy_model()", "after": ["train"]}
        ...     ]
        ... )
    """

    def __init__(self, session: Session, database: str, schema: str):
        """Initialize the task orchestrator.

        Args:
            session: Active Snowflake session to use for operations
            database: Target database for monitoring
            schema: Target schema for monitoring
        """
        self.session = session
        self.database = database
        self.schema = schema
        self.logger = get_logger(__name__)

    def create_task(
        self,
        name: str,
        sql: str,
        schedule: Optional[str] = None,
        warehouse: str = "COMPUTE_WH",
        after: Optional[List[str]] = None,
    ) -> bool:
        """Create a Snowflake Task.

        Args:
            name: Name of the task to create
            sql: SQL statement to execute
            schedule: Schedule for the task
            warehouse: Warehouse to use for the task
            after: List of tasks to run after

        Returns:
            True if task was created successfully, False otherwise
        """
        try:
            task_sql = f"CREATE OR REPLACE TASK {self.database}.{self.schema}.{name}\n"
            task_sql += f"WAREHOUSE = {warehouse}\n"

            if schedule:
                task_sql += f"SCHEDULE = '{schedule}'\n"
            elif after:
                task_sql += f"AFTER {', '.join(after)}\n"

            task_sql += f"AS\n{sql}"

            self.session.sql(task_sql).collect()
            self.logger.info(f"Created task: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create task {name}: {e}")
            return False

    def resume_task(self, name: str) -> bool:
        """Resume a task.

        Args:
            name: Name of the task to resume

        Returns:
            True if task was resumed successfully, False otherwise
        """
        try:
            self.session.sql(
                f"ALTER TASK {self.database}.{self.schema}.{name} RESUME"
            ).collect()
            return True
        except Exception as e:
            self.logger.error(f"Failed to resume task {name}: {e}")
            return False
