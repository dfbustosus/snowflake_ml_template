"""Deployment orchestrator for managing deployment operations."""

from typing import Any, Dict

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.deployment import (
    BaseDeploymentStrategy,
    DeploymentResult,
)
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class DeploymentOrchestrator:
    """Orchestrate model deployment operations.

    Example:
        >>> orchestrator = DeploymentOrchestrator(session)
        >>> orchestrator.register_strategy("udf", WarehouseUDFStrategy(config))
        >>> result = orchestrator.execute("udf")
    """

    def __init__(self, session: Session):
        """Initialize the orchestrator with a Snowflake session."""
        self.session = session
        self.strategies: Dict[str, BaseDeploymentStrategy] = {}
        self.logger = get_logger(__name__)

    def register_strategy(self, name: str, strategy: BaseDeploymentStrategy) -> None:
        """Register strategy for deployment."""
        # Set the session on the strategy if it has a set_session method
        if hasattr(strategy, "set_session"):
            strategy.set_session(self.session)
        self.strategies[name] = strategy
        self.logger.info(f"Registered deployment strategy: {name}")

    def execute(self, strategy_name: str) -> DeploymentResult:
        """Execute deployment strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")

        strategy = self.strategies[strategy_name]

        if not strategy.validate():
            raise ValueError(f"Strategy validation failed: {strategy_name}")

        result = strategy.deploy()

        self.logger.info(
            f"Deployment completed: {strategy_name}", extra={"status": result.status}
        )

        return result

    def health_check(self, strategy_name: str) -> Dict[str, Any]:
        """Check health of deployment strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")

        return self.strategies[strategy_name].health_check()
