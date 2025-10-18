"""Ingestion orchestrator for managing ingestion operations."""

from typing import Any, Dict, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    ExecutionEventTracker,
    IngestionResult,
)
from snowflake_ml_template.utils.logging import get_logger


class IngestionOrchestrator:
    """Manage registration and execution of ingestion strategies."""

    def __init__(
        self,
        session: Session,
        tracker: Optional[ExecutionEventTracker] = None,
    ) -> None:
        """Initialize the orchestrator."""
        if session is None:
            raise ValueError("Session cannot be None")
        self.session = session
        self.tracker = tracker
        self._logger = get_logger(__name__)
        self._strategies: Dict[str, BaseIngestionStrategy] = {}

    def register_strategy(self, name: str, strategy: BaseIngestionStrategy) -> None:
        """Register an ingestion strategy."""
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be a non-empty string")
        if not isinstance(strategy, BaseIngestionStrategy):
            raise TypeError("Strategy must inherit from BaseIngestionStrategy")

        strategy.set_session(self.session)
        strategy.set_tracker(self.tracker)
        self._strategies[name] = strategy
        self._logger.info(
            "Registered ingestion strategy",
            extra={"strategy_name": name, "method": strategy.config.method.value},
        )

    def unregister_strategy(self, name: str) -> None:
        """Unregister an ingestion strategy."""
        if name in self._strategies:
            del self._strategies[name]

    def get_strategy(self, name: str) -> BaseIngestionStrategy:
        """Get an ingestion strategy by name."""
        try:
            return self._strategies[name]
        except KeyError as exc:
            raise ValueError(f"Strategy not found: {name}") from exc

    @property
    def strategies(self) -> Dict[str, BaseIngestionStrategy]:
        """Return a shallow copy of registered strategies."""
        return dict(self._strategies)

    def execute(self, strategy_name: str, **kwargs: Any) -> IngestionResult:
        """Execute an ingestion strategy."""
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Strategy name must be a non-empty string")

        strategy = self.get_strategy(strategy_name)

        if not strategy.validate():
            raise ValueError(f"Strategy validation failed: {strategy_name}")

        config = strategy.config
        if not config or not config.source:
            raise ValueError("Invalid strategy configuration: missing source")

        result = strategy.execute_ingestion(
            source=config.source,
            target=strategy.get_target_table_name(),
            **kwargs,
        )

        self._logger.info(
            "Ingestion completed",
            extra={
                "strategy_name": strategy_name,
                "status": result.status,
                "rows_loaded": result.rows_loaded,
            },
        )
        return result
