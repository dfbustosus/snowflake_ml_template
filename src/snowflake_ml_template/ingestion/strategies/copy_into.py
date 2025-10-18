"""COPY INTO ingestion strategy for bulk data loading."""

from datetime import datetime
from typing import Any, Optional

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


class CopyIntoStrategy(BaseIngestionStrategy):
    """Bulk data ingestion using COPY INTO command.

    This strategy is optimal for:
    - Bulk data loads
    - Historical data backfills
    - Batch processing

    Example:
        >>> config = IngestionConfig(
        ...     method=IngestionMethod.COPY_INTO,
        ...     source=DataSource(source_type=SourceType.S3, location="s3://bucket/data/", file_format="CSV"),
        ...     target_database="ML_DEV_DB",
        ...     target_schema="RAW_DATA",
        ...     target_table="TRANSACTIONS",
        ...     warehouse="INGEST_WH"
        ... )
        >>> strategy = CopyIntoStrategy(config)
        >>> result = strategy.ingest(config.source, "ML_DEV_DB.RAW_DATA.TRANSACTIONS")
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Initialize the CopyIntoStrategy with an IngestionConfig.

        Args:
            config: Configuration for the COPY INTO ingestion
        """
        super().__init__(config)
        self._session: Optional[Session] = None

    def set_session(self, session: Session) -> None:
        """Set the Snowflake session for this strategy.

        Args:
            session: Active Snowflake session to use for operations

        Raises:
            ValueError: If session is None
        """
        if session is None:
            raise ValueError("Session cannot be None")
        self._session = session

    @property
    def session(self) -> Session:
        """Get the active Snowflake session.

        Returns:
            The active Snowflake session

        Raises:
            ValueError: If session is not set
        """
        if self._session is None:
            raise ValueError("Session is not set. Call set_session() first.")
        return self._session

    def ingest(self, source: DataSource, target: str, **kwargs: Any) -> IngestionResult:
        """Execute COPY INTO for bulk data loading.

        Args:
            source: The data source to ingest from
            target: The target table to load data into
            **kwargs: Additional keyword arguments

        Returns:
            IngestionResult containing the result of the operation
        """
        start_time = datetime.utcnow()

        try:
            sql = f"""
            COPY INTO {target}
            FROM '{source.location}'
            FILE_FORMAT = (TYPE = '{source.file_format}')
            ON_ERROR = '{self.config.on_error}'
            PURGE = {self.config.purge}
            """

            result = self.session.sql(sql).collect()
            rows_loaded = sum(row["rows_loaded"] for row in result)
            files_processed = len(result)

            return IngestionResult(
                status="success",
                method=IngestionMethod.COPY_INTO,
                target_table=target,
                rows_loaded=rows_loaded,
                files_processed=files_processed,
                start_time=start_time,
                end_time=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"COPY INTO failed: {e}")
            return IngestionResult(
                status="failed",
                method=IngestionMethod.COPY_INTO,
                target_table=target,
                start_time=start_time,
                end_time=datetime.utcnow(),
                error=str(e),
            )

    def validate(self) -> bool:
        """Validate configuration."""
        return bool(self.config.source.location and self.config.target_table)
