"""Base abstractions for data ingestion strategies.

This module defines the Strategy pattern for data ingestion. Different
ingestion methods (Snowpipe, COPY INTO, Streaming) implement the same
interface, allowing them to be used interchangeably.

Classes:
    DataSource: Configuration for a data source
    IngestionConfig: Configuration for ingestion operation
    IngestionResult: Result of ingestion operation
    BaseIngestionStrategy: Abstract base class for ingestion strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from snowflake_ml_template.utils.logging import StructuredLogger, get_logger

if TYPE_CHECKING:
    from snowflake.snowpark import Session


class IngestionMethod(Enum):
    """Enumeration of supported ingestion methods."""

    SNOWPIPE = "snowpipe"
    COPY_INTO = "copy_into"
    STREAMING = "streaming"
    EXTERNAL_TABLE = "external_table"


class SourceType(Enum):
    """Enumeration of supported source types."""

    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    KAFKA = "kafka"
    REST_API = "rest_api"
    LOCAL_FILE = "local_file"


@dataclass
class DataSource:
    """Configuration for a data source.

    Attributes:
        source_type: Type of data source (S3, Azure, GCS, etc.)
        location: URI or path to the data source
        file_format: Snowflake file format (CSV, JSON, PARQUET, etc.)
        credentials: Optional credentials for accessing the source
        pattern: Optional file pattern for filtering (e.g., "*.csv")
        metadata: Additional source-specific metadata
    """

    source_type: SourceType
    location: str
    file_format: str
    credentials: Optional[Dict[str, str]] = None
    pattern: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data source configuration."""
        if not self.location:
            raise ValueError("Data source location cannot be empty")
        if not self.file_format:
            raise ValueError("File format cannot be empty")


@dataclass
class IngestionConfig:
    """Configuration for data ingestion operation.

    Attributes:
        method: Ingestion method to use
        source: Data source configuration
        target_database: Target Snowflake database
        target_schema: Target Snowflake schema
        target_table: Target table name
        warehouse: Snowflake warehouse for ingestion
        on_error: Error handling strategy (CONTINUE, SKIP_FILE, ABORT_STATEMENT)
        purge: Whether to purge files after successful ingestion
        force: Whether to force reload of files
        validation_mode: Validation mode (RETURN_ERRORS, RETURN_ALL_ERRORS, etc.)
        metadata: Additional ingestion-specific metadata
    """

    method: IngestionMethod
    source: DataSource
    target_database: str
    target_schema: str
    target_table: str
    warehouse: str
    on_error: str = "ABORT_STATEMENT"
    purge: bool = False
    force: bool = False
    validation_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ingestion configuration."""
        if not self.target_database:
            raise ValueError("Target database cannot be empty")
        if not self.target_schema:
            raise ValueError("Target schema cannot be empty")
        if not self.target_table:
            raise ValueError("Target table cannot be empty")
        if not self.warehouse:
            raise ValueError("Warehouse cannot be empty")

        valid_on_error = ["CONTINUE", "SKIP_FILE", "ABORT_STATEMENT"]
        if self.on_error not in valid_on_error:
            raise ValueError(
                f"Invalid on_error value: {self.on_error}. "
                f"Must be one of: {valid_on_error}"
            )


@dataclass
class IngestionResult:
    """Result of data ingestion operation.

    Attributes:
        status: Ingestion status (success, failed, partial)
        method: Ingestion method used
        target_table: Fully qualified target table name
        rows_loaded: Number of rows successfully loaded
        rows_failed: Number of rows that failed to load
        files_processed: Number of files processed
        start_time: When ingestion started
        end_time: When ingestion completed
        duration_seconds: Total ingestion time in seconds
        error: Error message if ingestion failed
        metadata: Additional result metadata (e.g., Snowpipe ID, COPY ID)
    """

    status: str
    method: IngestionMethod
    target_table: str
    rows_loaded: int = 0
    rows_failed: int = 0
    files_processed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate duration if start and end times are provided."""
        if self.start_time and self.end_time:
            # Normalize timezone awareness to avoid TypeError on subtraction
            s = self.start_time
            e = self.end_time
            if (s.tzinfo is None) != (e.tzinfo is None):
                # Convert both to naive UTC for consistency
                if s.tzinfo is not None:
                    s = s.replace(tzinfo=None)
                if e.tzinfo is not None:
                    e = e.replace(tzinfo=None)
            delta = e - s
            self.duration_seconds = delta.total_seconds()


class BaseIngestionStrategy(ABC):
    """Abstract base class for data ingestion strategies.

    This class defines the interface that all ingestion strategies must
    implement. It follows the Strategy design pattern, allowing different
    ingestion methods to be used interchangeably.

    Subclasses must implement:
    - ingest(): Perform the actual data ingestion
    - validate(): Validate the ingestion configuration

    Attributes:
        config: Ingestion configuration
        logger: Logger instance for structured logging

    Example:
        >>> class SnowpipeStrategy(BaseIngestionStrategy):
        ...     def ingest(self, source: DataSource, target: str) -> IngestionResult:
        ...         # Implement Snowpipe ingestion
        ...         pass
        >>>
        >>> strategy = SnowpipeStrategy(config)
        >>> result = strategy.ingest(source, target)
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Initialize the ingestion strategy.

        Args:
            config: Ingestion configuration

        Raises:
            ValueError: If config is None
        """
        if config is None:
            raise ValueError("Config cannot be None")

        self.config = config
        self.logger: StructuredLogger = self._get_logger()
        # Internal session handle; subclasses may provide their own property wrappers
        self._session: Optional["Session"] = None

    def _get_logger(self) -> StructuredLogger:
        """Return a structured logger scoped to the ingestion strategy."""
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(logger_name)

    @abstractmethod
    def ingest(self, source: DataSource, target: str, **kwargs: Any) -> IngestionResult:
        """Ingest data from source to target.

        This is the main method that performs data ingestion. Subclasses
        must implement this method with their specific ingestion logic.

        Args:
            source: Data source configuration
            target: Fully qualified target table name
            **kwargs: Additional ingestion-specific parameters

        Returns:
            IngestionResult with status and metadata

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement ingest()")

    @abstractmethod
    def validate(self) -> bool:
        """Validate ingestion configuration.

        This method validates that the ingestion configuration is correct
        and all required resources (source, target, credentials) are available.

        Returns:
            True if configuration is valid, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )

    def get_target_table_name(self) -> str:
        """Get fully qualified target table name.

        Returns:
            Fully qualified table name (DATABASE.SCHEMA.TABLE)
        """
        return (
            f"{self.config.target_database}."
            f"{self.config.target_schema}."
            f"{self.config.target_table}"
        )

    def set_session(self, session: "Session") -> None:
        """Set the Snowflake session for this strategy.

        Args:
            session: Active Snowflake session to use for operations
        """
        # Avoid colliding with subclasses that define a read-only 'session' property
        self._session = session
