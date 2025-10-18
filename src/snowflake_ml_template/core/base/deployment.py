"""Base abstractions for model deployment strategies.

This module defines the interface for model deployment. Different deployment
strategies (Warehouse UDF, SPCS, External) implement the same interface,
allowing them to be used interchangeably.

Classes:
    DeploymentStrategy: Enum of deployment strategies
    DeploymentTarget: Enum of deployment targets
    DeploymentConfig: Configuration for deployment operation
    DeploymentResult: Result of deployment operation
    BaseDeploymentStrategy: Abstract base class for deployment strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from snowflake_ml_template.core.base.tracking import (
    ExecutionEventTracker,
    emit_tracker_event,
)
from snowflake_ml_template.utils.logging import StructuredLogger, get_logger


class DeploymentStrategy(Enum):
    """Enumeration of supported deployment strategies."""

    WAREHOUSE_UDF = "warehouse_udf"
    SPCS = "spcs"
    EXTERNAL = "external"


class DeploymentTarget(Enum):
    """Enumeration of deployment targets."""

    BATCH = "batch"
    REALTIME = "realtime"
    STREAMING = "streaming"


class DeploymentStatus(Enum):
    """Enumeration of deployment outcomes."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment operation.

    Attributes:
        strategy: Deployment strategy to use
        target: Deployment target (batch, realtime, streaming)
        model_name: Name of the model to deploy
        model_version: Version of the model to deploy
        model_artifact_path: Path to model artifact
        deployment_database: Target database for deployment
        deployment_schema: Target schema for deployment
        deployment_name: Name for the deployed model/service
        warehouse: Snowflake warehouse for deployment
        compute_pool: Compute pool for SPCS deployment
        instance_count: Number of instances for SPCS
        gpu_enabled: Whether to use GPU instances
        max_batch_size: Maximum batch size for inference
        timeout_seconds: Timeout for inference requests
        enable_monitoring: Whether to enable monitoring
        metadata: Additional deployment-specific metadata
    """

    strategy: DeploymentStrategy
    target: DeploymentTarget
    model_name: str
    model_version: str
    model_artifact_path: str
    deployment_database: str
    deployment_schema: str
    deployment_name: str
    warehouse: str
    compute_pool: Optional[str] = None
    instance_count: int = 1
    gpu_enabled: bool = False
    max_batch_size: int = 1000
    timeout_seconds: int = 300
    enable_monitoring: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate deployment configuration."""
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not self.model_version:
            raise ValueError("Model version cannot be empty")
        if not self.model_artifact_path:
            raise ValueError("Model artifact path cannot be empty")
        if not self.deployment_database:
            raise ValueError("Deployment database cannot be empty")
        if not self.deployment_name:
            raise ValueError("Deployment name cannot be empty")

        if self.strategy == DeploymentStrategy.SPCS and not self.compute_pool:
            raise ValueError("Compute pool is required for SPCS deployment")

        if self.instance_count < 1:
            raise ValueError(f"Instance count must be >= 1, got {self.instance_count}")


@dataclass
class DeploymentResult:
    """Result of model deployment operation.

    Attributes:
        status: Deployment status (success, failed)
        strategy: Deployment strategy used
        target: Deployment target
        deployment_name: Name of deployed model/service
        endpoint_url: URL endpoint for inference (if applicable)
        udf_name: UDF name (for warehouse deployment)
        service_name: Service name (for SPCS deployment)
        start_time: When deployment started
        end_time: When deployment completed
        duration_seconds: Total deployment time in seconds
        error: Error message if deployment failed
        metadata: Additional result metadata
    """

    status: str | DeploymentStatus
    strategy: DeploymentStrategy
    target: DeploymentTarget
    deployment_name: str
    endpoint_url: Optional[str] = None
    udf_name: Optional[str] = None
    service_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate duration if start and end times are provided."""
        if isinstance(self.status, DeploymentStatus):
            self.status = self.status.value

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

    @property
    def deployment_status(self) -> DeploymentStatus:
        """Return deployment status as an enum."""
        if isinstance(self.status, DeploymentStatus):
            return self.status

        try:
            return DeploymentStatus(self.status)
        except ValueError:
            if self.error:
                return DeploymentStatus.FAILED
            return (
                DeploymentStatus.PARTIAL if self.metrics else DeploymentStatus.SUCCESS
            )


class BaseDeploymentStrategy(ABC):
    """Abstract base class for model deployment strategies.

    This class defines the interface that all deployment strategies must
    implement. It follows the Strategy design pattern, allowing different
    deployment methods to be used interchangeably.

    Subclasses must implement:
    - deploy(): Deploy the model
    - undeploy(): Remove the deployment
    - validate(): Validate the deployment configuration
    - health_check(): Check deployment health

    Attributes:
        config: Deployment configuration
        logger: Logger instance for structured logging

    Example:
        >>> class WarehouseUDFStrategy(BaseDeploymentStrategy):
        ...     def deploy(self, **kwargs) -> DeploymentResult:
        ...         # Implement UDF deployment
        ...         pass
        >>>
        >>> strategy = WarehouseUDFStrategy(config)
        >>> result = strategy.deploy()
    """

    def __init__(
        self,
        config: DeploymentConfig,
        tracker: ExecutionEventTracker | None = None,
    ) -> None:
        """Initialize the deployment strategy.

        Args:
            config: Deployment configuration
            tracker: Optional execution tracker for telemetry events

        Raises:
            ValueError: If config is None
        """
        if config is None:
            raise ValueError("Config cannot be None")

        self.config = config
        self.logger: StructuredLogger = self._get_logger()
        self._tracker = tracker

    def _get_logger(self) -> StructuredLogger:
        """Return a structured logger scoped to the deployment strategy."""
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(logger_name)

    @property
    def tracker(self) -> ExecutionEventTracker | None:
        """Return the configured execution tracker."""
        return self._tracker

    def _emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit a deployment-related event."""
        base_payload: Dict[str, Any] = {
            "strategy": self.config.strategy.value,
            "target": self.config.target.value,
            "deployment_name": self.config.deployment_name,
        }
        base_payload.update(payload)
        emit_tracker_event(
            tracker=self._tracker,
            component=self.__class__.__name__,
            event=event,
            payload=base_payload,
        )

    @abstractmethod
    def deploy(self, **kwargs: Any) -> DeploymentResult:
        """Deploy the model.

        This is the main method that performs model deployment. Subclasses
        must implement this method with their specific deployment logic.

        Args:
            **kwargs: Additional deployment-specific parameters

        Returns:
            DeploymentResult with status and metadata

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement deploy()")

    def pre_deploy(self, **kwargs: Any) -> None:
        """Execute hook before deployment begins."""
        pass

    def post_deploy(self, result: DeploymentResult) -> None:
        """Execute hook after successful deployment."""
        pass

    def on_deploy_error(self, error: Exception) -> None:
        """Handle deployment failure in hook."""
        pass

    @abstractmethod
    def undeploy(self) -> bool:
        """Remove the deployment.

        This method removes the deployed model/service.

        Returns:
            True if undeployment was successful, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement undeploy()"
        )

    def pre_undeploy(self) -> None:
        """Execute hook before undeployment."""
        pass

    def post_undeploy(self, success: bool) -> None:
        """Execute hook after undeployment completes."""
        pass

    def on_undeploy_error(self, error: Exception) -> None:
        """Handle undeployment failure in hook."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate deployment configuration.

        This method validates that the deployment configuration is correct
        and all required resources are available.

        Returns:
            True if configuration is valid, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check deployment health.

        This method checks if the deployed model/service is healthy
        and ready to serve requests.

        Returns:
            Dictionary with health status and metrics

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement health_check()"
        )

    def pre_health_check(self) -> None:
        """Execute hook before health check."""
        pass

    def post_health_check(self, status: Dict[str, Any]) -> None:
        """Execute hook after health check completes."""
        pass

    def on_health_check_error(self, error: Exception) -> None:
        """Handle health-check failure in hook."""
        pass

    def get_deployment_full_name(self) -> str:
        """Get fully qualified deployment name.

        Returns:
            Fully qualified name (DATABASE.SCHEMA.DEPLOYMENT_NAME)
        """
        return (
            f"{self.config.deployment_database}."
            f"{self.config.deployment_schema}."
            f"{self.config.deployment_name}"
        )

    def execute_deploy(self, **kwargs: Any) -> DeploymentResult:
        """Execute deployment with lifecycle hooks and tracking."""
        self._emit_event(
            event="deployment_start",
            payload={"timestamp": datetime.now(timezone.utc).isoformat()},
        )
        self.pre_deploy(**kwargs)
        start_time = datetime.now(timezone.utc)

        try:
            result = self.deploy(**kwargs)
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.metrics.setdefault("duration_seconds", duration)
            result.start_time = result.start_time or start_time
            result.end_time = result.end_time or datetime.now(timezone.utc)
            self.post_deploy(result)
            self._emit_event(
                event="deployment_end",
                payload={
                    "status": result.status,
                    "metrics": result.metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return result
        except Exception as exc:
            self.logger.error("Deployment failed", exc_info=True)
            self.on_deploy_error(exc)
            self._emit_event(
                event="deployment_error",
                payload={
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise

    def execute_undeploy(self) -> bool:
        """Execute undeployment with hooks and tracking."""
        self._emit_event(
            event="undeploy_start",
            payload={"timestamp": datetime.now(timezone.utc).isoformat()},
        )
        self.pre_undeploy()

        try:
            success = self.undeploy()
            self.post_undeploy(success)
            self._emit_event(
                event="undeploy_end",
                payload={
                    "success": success,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return success
        except Exception as exc:
            self.logger.error("Undeploy failed", exc_info=True)
            self.on_undeploy_error(exc)
            self._emit_event(
                event="undeploy_error",
                payload={
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise

    def execute_health_check(self) -> Dict[str, Any]:
        """Execute health check with hooks and tracking."""
        self._emit_event(
            event="health_check_start",
            payload={"timestamp": datetime.now(timezone.utc).isoformat()},
        )
        self.pre_health_check()
        try:
            status = self.health_check()
            self.post_health_check(status)
            self._emit_event(
                event="health_check_end",
                payload={
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return status
        except Exception as exc:
            self.logger.error("Health check failed", exc_info=True)
            self.on_health_check_error(exc)
            self._emit_event(
                event="health_check_error",
                payload={
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
