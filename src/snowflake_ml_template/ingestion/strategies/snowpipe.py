"""Snowpipe ingestion strategy for continuous data loading."""

from datetime import datetime, timezone
from typing import Any

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    DataSource,
    IngestionConfig,
    IngestionMethod,
    IngestionResult,
)
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class SnowpipeStrategy(BaseIngestionStrategy):
    """Continuous data ingestion using Snowpipe.

    Optimal for:
    - Real-time/near-real-time ingestion
    - Event-driven data loading
    - Micro-batch processing
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Initialize the SnowpipeStrategy with an IngestionConfig.

        Args:
            config: Configuration for the Snowpipe ingestion
        """
        super().__init__(config)

    def set_session(self, session: Session) -> None:
        """Set the Snowflake session for this strategy."""
        super().set_session(session)

    def ingest(self, source: DataSource, target: str, **kwargs: Any) -> IngestionResult:
        """Create or refresh Snowpipe."""
        if self.session is None:
            raise ValueError("Session is not set. Call set_session() first.")

        start_time = datetime.now(timezone.utc)

        try:
            pipe_name = f"{target}_PIPE"

            sql = f"""
            CREATE OR REPLACE PIPE {pipe_name}
            AUTO_INGEST = TRUE
            AS
            COPY INTO {target}
            FROM '{source.location}'
            FILE_FORMAT = (TYPE = '{source.file_format}')
            """

            self.session.sql(sql).collect()

            return IngestionResult(
                status="success",
                method=IngestionMethod.SNOWPIPE,
                target_table=target,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                metadata={"pipe_name": pipe_name},
            )
        except Exception as e:
            logger.error(f"Snowpipe creation failed: {e}")
            return IngestionResult(
                status="failed",
                method=IngestionMethod.SNOWPIPE,
                target_table=target,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error=str(e),
            )

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            bool: True if configuration is valid, False otherwise

        Raises:
            ValueError: If required configuration is missing
        """
        if not self.config:
            return False

        if not self.config.source or not self.config.source.location:
            return False

        if not self.config.target_table:
            return False

        return True

    def get_target_table_name(self) -> str:
        """Get the target table name from configuration.

        Returns:
            str: The target table name

        Raises:
            ValueError: If target table is not configured
        """
        if not self.config or not self.config.target_table:
            raise ValueError("Target table is not configured")
        return self.config.target_table
