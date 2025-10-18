"""Ingestion orchestrator for managing ingestion operations."""

from typing import TYPE_CHECKING, Dict

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.ingestion import IngestionResult

if TYPE_CHECKING:
    from snowflake_ml_template.core.base.ingestion import BaseIngestionStrategy

from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class IngestionOrchestrator:
    """Orchestrate data ingestion operations.

    This class manages ingestion strategies and executes ingestion operations
    following the Strategy pattern.

    Example:
        >>> orchestrator = IngestionOrchestrator(session)
        >>> orchestrator.register_strategy("copy_into", CopyIntoStrategy(config))
        >>> result = orchestrator.execute("copy_into")
    """

    def __init__(self, session: Session) -> None:
        """Initialize the IngestionOrchestrator with a Snowflake session.

        Args:
            session: Active Snowflake session for database operations
        """
        if session is None:
            raise ValueError("Session cannot be None")
        self.session = session
        self.strategies: Dict[str, BaseIngestionStrategy] = {}
        self.logger = get_logger(__name__)

    def register_strategy(self, name: str, strategy: "BaseIngestionStrategy") -> None:
        """Register an ingestion strategy.

        Args:
            name: Name to register the strategy under
            strategy: Strategy instance to register

        Raises:
            ValueError: If name or strategy is invalid
        """
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be a non-empty string")
        if not hasattr(strategy, "set_session"):
            raise ValueError("Strategy must have a set_session method")
        if not hasattr(strategy, "ingest"):
            raise ValueError("Strategy must have an ingest method")

        # Use setattr to bypass mypy's attribute check since we've verified it exists
        getattr(strategy, "set_session")(self.session)
        self.strategies[name] = strategy
        self.logger.info("Registered ingestion strategy", extra={"strategy_name": name})

    def execute(self, strategy_name: str) -> IngestionResult:
        """Execute an ingestion strategy.

        Args:
            strategy_name: Name of the strategy to execute

        Returns:
            IngestionResult containing the result of the operation

        Raises:
            ValueError: If strategy is not found or validation fails
        """
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Strategy name must be a non-empty string")

        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")

        strategy = self.strategies[strategy_name]

        if not strategy.validate():
            raise ValueError(f"Strategy validation failed: {strategy_name}")

        target = strategy.get_target_table_name()
        if not strategy.config or not strategy.config.source:
            raise ValueError("Invalid strategy configuration: missing source")

        result = strategy.ingest(strategy.config.source, target)

        self.logger.info(
            "Ingestion completed",
            extra={
                "strategy_name": strategy_name,
                "status": result.status,
                "rows_loaded": result.rows_loaded,
            },
        )

        return result
