"""Base abstractions for data transformation strategies.

This module defines the interface for data transformations. Transformations
can be implemented using Snowpark DataFrames, SQL, or dbt, all following
the same interface.

Classes:
    TransformationConfig: Configuration for transformation operation
    TransformationResult: Result of transformation operation
    BaseTransformation: Abstract base class for transformations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from snowflake_ml_template.utils.logging import StructuredLogger, get_logger


class TransformationType(Enum):
    """Enumeration of supported transformation types."""

    SNOWPARK = "snowpark"
    SQL = "sql"
    DBT = "dbt"
    DYNAMIC_TABLE = "dynamic_table"


@dataclass
class TransformationConfig:
    """Configuration for data transformation operation.

    Attributes:
        transformation_type: Type of transformation
        source_database: Source Snowflake database
        source_schema: Source Snowflake schema
        source_table: Source table name
        target_database: Target Snowflake database
        target_schema: Target Snowflake schema
        target_table: Target table name
        warehouse: Snowflake warehouse for transformation
        mode: Write mode (overwrite, append, merge)
        partition_by: Optional partition columns
        cluster_by: Optional clustering columns
        metadata: Additional transformation-specific metadata
    """

    transformation_type: TransformationType
    source_database: str
    source_schema: str
    source_table: str
    target_database: str
    target_schema: str
    target_table: str
    warehouse: str
    mode: str = "overwrite"
    partition_by: Optional[List[str]] = None
    cluster_by: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate transformation configuration."""
        if not self.source_database:
            raise ValueError("Source database cannot be empty")
        if not self.target_database:
            raise ValueError("Target database cannot be empty")
        if not self.warehouse:
            raise ValueError("Warehouse cannot be empty")

        valid_modes = ["overwrite", "append", "merge"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be one of: {valid_modes}"
            )


@dataclass
class TransformationResult:
    """Result of data transformation operation.

    Attributes:
        status: Transformation status (success, failed)
        transformation_type: Type of transformation used
        target_table: Fully qualified target table name
        rows_processed: Number of rows processed
        rows_written: Number of rows written to target
        start_time: When transformation started
        end_time: When transformation completed
        duration_seconds: Total transformation time in seconds
        error: Error message if transformation failed
        metadata: Additional result metadata
    """

    status: str
    transformation_type: TransformationType
    target_table: str
    rows_processed: int = 0
    rows_written: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate duration if start and end times are provided."""
        if self.start_time and self.end_time:
            start = self.start_time
            end = self.end_time

            if (start.tzinfo is None) != (end.tzinfo is None):
                if start.tzinfo is not None:
                    start = start.replace(tzinfo=None)
                if end.tzinfo is not None:
                    end = end.replace(tzinfo=None)

            delta = end - start
            self.duration_seconds = delta.total_seconds()


class BaseTransformation(ABC):
    """Abstract base class for data transformations.

    This class defines the interface that all transformation implementations
    must follow. It supports different transformation engines (Snowpark, SQL,
    dbt) through a common interface.

    Attributes:
        config: Transformation configuration
        logger: Logger instance for structured logging

    Example:
        >>> class SnowparkTransformation(BaseTransformation):
        ...     def transform(self, **kwargs) -> TransformationResult:
        ...         # Implement Snowpark transformation
        ...         pass
        >>>
        >>> transformation = SnowparkTransformation(config)
        >>> result = transformation.transform()
    """

    def __init__(self, config: TransformationConfig) -> None:
        """Initialize the transformation.

        Args:
            config: Transformation configuration

        Raises:
            ValueError: If config is None
        """
        if config is None:
            raise ValueError("Config cannot be None")

        self.config = config
        self.logger: StructuredLogger = self._get_logger()

    def _get_logger(self) -> StructuredLogger:
        """Return a structured logger scoped to the transformation."""
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(logger_name)

    @abstractmethod
    def transform(self, **kwargs: Any) -> TransformationResult:
        """Execute the transformation.

        This is the main method that performs data transformation. Subclasses
        must implement this method with their specific transformation logic.

        Args:
            **kwargs: Additional transformation-specific parameters

        Returns:
            TransformationResult with status and metadata

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform()"
        )

    @abstractmethod
    def validate(self) -> bool:
        """Validate transformation configuration.

        This method validates that the transformation configuration is correct
        and all required resources are available.

        Returns:
            True if configuration is valid, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )

    def get_source_table_name(self) -> str:
        """Get fully qualified source table name.

        Returns:
            Fully qualified table name (DATABASE.SCHEMA.TABLE)
        """
        return (
            f"{self.config.source_database}."
            f"{self.config.source_schema}."
            f"{self.config.source_table}"
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
