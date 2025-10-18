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
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, cast

from snowflake_ml_template.core.base.tracking import (
    ExecutionEventTracker,
    emit_tracker_event,
)
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


class IngestionStatus(Enum):
    """Enumeration of ingestion outcomes."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class CopyIntoOptions:
    """Optional COPY INTO configuration overrides."""

    file_format_name: Optional[str] = None
    files: Optional[List[str]] = None
    pattern: Optional[str] = None
    force: Optional[bool] = None
    on_error: Optional[str] = None
    size_limit: Optional[int] = None
    max_files: Optional[int] = None
    parallel: Optional[int] = None
    enable_octal: Optional[bool] = None
    return_failed_only: bool = False
    disable_notification: bool = False
    enforce_length: Optional[bool] = None
    truncate_columns: Optional[bool] = None
    trim_space: Optional[bool] = None
    null_if: Optional[List[str]] = None


@dataclass
class SnowpipeNotificationChannel:
    """Notification channel settings for auto-ingest Snowpipe."""

    integration: str
    channel_arn: str
    prefix: Optional[str] = None
    suffix: Optional[str] = None


class SnowpipeLoadMethod(Enum):
    """Supported Snowpipe load trigger mechanisms."""

    AUTO_INGEST = "auto_ingest"
    REST_API = "rest_api"
    STREAMING = "streaming"


@dataclass
class SnowpipeOptions:
    """Optional Snowpipe configuration overrides."""

    pipe_name: Optional[str] = None
    auto_ingest: bool = False
    notification_channel: Optional[SnowpipeNotificationChannel] = None
    integration: Optional[str] = None
    rest_endpoint: Optional[str] = None
    role: Optional[str] = None
    load_method: SnowpipeLoadMethod = SnowpipeLoadMethod.AUTO_INGEST
    warehouse: Optional[str] = None
    comment: Optional[str] = None
    file_format_name: Optional[str] = None
    pattern: Optional[str] = None
    error_integration: Optional[str] = None
    dead_letter_queue: Optional[str] = None


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
        copy_history_window_minutes: Optional window for COPY_HISTORY lookups
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
    copy_options: Optional["CopyIntoOptions"] = None
    snowpipe_options: Optional["SnowpipeOptions"] = None
    copy_history_window_minutes: Optional[int] = None

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

        if (
            self.copy_history_window_minutes is not None
            and self.copy_history_window_minutes < 0
        ):
            raise ValueError("copy_history_window_minutes must be non-negative")


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

    status: str | IngestionStatus
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
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate duration if start and end times are provided."""
        if isinstance(self.status, IngestionStatus):
            self.status = self.status.value

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

    @property
    def ingestion_status(self) -> IngestionStatus:
        """Return status as an enum."""
        if isinstance(self.status, IngestionStatus):
            return self.status

        try:
            return IngestionStatus(self.status)
        except ValueError:
            if self.error:
                return IngestionStatus.FAILED
            if self.rows_loaded:
                return IngestionStatus.PARTIAL
            return IngestionStatus.SUCCESS


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

    def __init__(
        self,
        config: IngestionConfig,
        tracker: ExecutionEventTracker | None = None,
    ) -> None:
        """Initialize the ingestion strategy.

        Args:
            config: Ingestion configuration
            tracker: Optional execution tracker for telemetry events

        Raises:
            ValueError: If config is None
        """
        if config is None:
            raise ValueError("Config cannot be None")

        self.config = config
        self.logger: StructuredLogger = self._get_logger()
        # Internal session handle; subclasses may provide their own property wrappers
        self._session: Optional["Session"] = None
        self._tracker = tracker

    def _get_logger(self) -> StructuredLogger:
        """Return a structured logger scoped to the ingestion strategy."""
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(logger_name)

    @property
    def tracker(self) -> ExecutionEventTracker | None:
        """Return the configured execution tracker."""
        return self._tracker

    @property
    def session(self) -> "Session":
        """Access the configured Snowflake session.

        Raises:
            RuntimeError: If a session has not been provided.
        """
        if self._session is None:
            raise RuntimeError(
                "Snowflake session has not been set. Call set_session() before accessing."
            )
        return self._session

    @session.setter
    def session(self, value: "Session") -> None:
        """Assign the Snowflake session and emit telemetry."""
        self._session = value
        self._emit_event(event="session_bound", payload={})

    def _emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit an ingestion-related event."""
        base_payload: Dict[str, Any] = {
            "method": self.config.method.value,
            "target": self.get_target_table_name(),
        }
        base_payload.update(payload)
        emit_tracker_event(
            tracker=self._tracker,
            component=self.__class__.__name__,
            event=event,
            payload=base_payload,
        )

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        if isinstance(row, dict):
            return row
        if isinstance(row, Mapping):
            return dict(row)
        if hasattr(row, "as_dict"):
            mapping = cast(Mapping[str, Any], row.as_dict())
            return dict(mapping)
        if hasattr(row, "_asdict"):
            mapping = cast(Mapping[str, Any], row._asdict())
            return dict(mapping)
        if hasattr(row, "__dict__") and isinstance(row.__dict__, dict):
            return dict(row.__dict__)
        return {}

    def _fetch_copy_history(
        self, target: str, window_minutes: int
    ) -> List[Dict[str, Any]]:
        try:
            history_sql = (
                "SELECT * FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY("
                f"TABLE_NAME => '{target}', "
                f"START_TIME => DATEADD('minute', -{window_minutes}, CURRENT_TIMESTAMP()))) "
                "ORDER BY LAST_LOAD_TIME DESC LIMIT 20"
            )
            rows = self.session.sql(history_sql).collect()
            return [self._row_to_dict(row) for row in rows if row is not None]
        except Exception:
            return []

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

    def pre_ingest(self, source: DataSource) -> None:
        """Execute hook before ingestion begins."""
        pass

    def post_ingest(self, result: IngestionResult) -> None:
        """Execute hook after successful ingestion."""
        pass

    def on_ingest_error(self, error: Exception) -> None:
        """Handle ingestion failure in hook."""
        pass

    def pre_validation(self, source: DataSource) -> None:
        """Perform governance checks before validating the source."""
        pass

    def validate_source(self, source: DataSource) -> Dict[str, Any]:
        """Return validation report for the source prior to ingestion."""
        return {}

    def post_validation(self, source: DataSource, report: Dict[str, Any]) -> None:
        """Handle governance reporting after source validation."""
        pass

    def on_validation_error(self, source: DataSource, error: Exception) -> None:
        """React to validation governance failures."""
        pass

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
        self.session = session

    def set_tracker(self, tracker: ExecutionEventTracker | None) -> None:
        """Attach an execution tracker to the strategy."""
        self._tracker = tracker

    def execute_ingestion(
        self, source: DataSource, target: str, **kwargs: Any
    ) -> IngestionResult:
        """Execute ingestion with lifecycle hooks and tracking."""
        history_window_config = self.config.copy_history_window_minutes
        history_window = kwargs.pop("history_window_minutes", history_window_config)
        history_window = int(history_window) if history_window is not None else 0
        try:
            self.pre_validation(source)
            validation_report = self.validate_source(source)
            self.post_validation(source, validation_report)
        except Exception as validation_error:
            self.on_validation_error(source, validation_error)
            raise

        self._emit_event(
            event="ingestion_start",
            payload={
                "source": source.location,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.pre_ingest(source)
        start_time = datetime.now(timezone.utc)

        try:
            result = self.ingest(source=source, target=target, **kwargs)
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.metrics.setdefault("duration_seconds", duration)
            result.start_time = result.start_time or start_time
            result.end_time = result.end_time or datetime.now(timezone.utc)
            self.post_ingest(result)
            self._emit_event(
                event="ingestion_end",
                payload={
                    "status": result.status,
                    "metrics": result.metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            if history_window > 0:
                history_records = self._fetch_copy_history(target, history_window)
                if history_records:
                    result.metadata.setdefault("copy_history", history_records)
            return result
        except Exception as exc:
            self.logger.exception("Ingestion failed")
            self.on_ingest_error(exc)
            failure_time = datetime.now(timezone.utc)
            failure_result = IngestionResult(
                status=IngestionStatus.FAILED.value,
                method=self.config.method,
                target_table=target,
                start_time=start_time,
                end_time=failure_time,
                error=str(exc),
                metadata={"exception_type": exc.__class__.__name__},
            )
            failure_result.metrics["duration_seconds"] = (
                failure_time - start_time
            ).total_seconds()
            self._emit_event(
                event="ingestion_error",
                payload={
                    "error": str(exc),
                    "timestamp": failure_time.isoformat(),
                },
            )
            return failure_result
