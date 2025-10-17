"""SPCS deployment strategy for real-time inference."""

from datetime import datetime
from typing import Any, Dict

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.deployment import (
    BaseDeploymentStrategy,
    DeploymentConfig,
    DeploymentResult,
    DeploymentStrategy,
)
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class SPCSStrategy(BaseDeploymentStrategy):
    """Deploy model to Snowpark Container Services for real-time inference.

    Optimal for:
    - Real-time inference (<200ms P95 latency)
    - GPU-accelerated models
    - Large models (>5GB)
    - Custom dependencies

    Example:
        >>> config = DeploymentConfig(
        ...     strategy=DeploymentStrategy.SPCS,
        ...     target=DeploymentTarget.REALTIME,
        ...     model_name="fraud_detector",
        ...     model_version="1.0.0",
        ...     model_artifact_path="@ML_MODELS_STAGE/fraud_v1.joblib",
        ...     deployment_database="ML_PROD_DB",
        ...     deployment_schema="MODELS",
        ...     deployment_name="fraud_service",
        ...     warehouse="INFERENCE_WH",
        ...     compute_pool="ML_GPU_POOL",
        ...     instance_count=2
        ... )
        >>> strategy = SPCSStrategy(config)
        >>> result = strategy.deploy()
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize the SPCS strategy."""
        super().__init__(config)
        self.session: Session = None

    def set_session(self, session: Session) -> None:
        """Set the Snowflake session."""
        self.session = session

    def deploy(self, **kwargs: Any) -> DeploymentResult:
        """Deploy model to SPCS."""
        start_time = datetime.utcnow()

        try:
            service_name = self.get_deployment_full_name()

            # Create service specification
            spec = f"""
            spec:
              containers:
              - name: inference
                image: /ml_models/inference:latest
                env:
                  MODEL_PATH: {self.config.model_artifact_path}
                  MODEL_NAME: {self.config.model_name}
                  MODEL_VERSION: {self.config.model_version}
                resources:
                  requests:
                    memory: 4Gi
                    cpu: 2
                  limits:
                    memory: 8Gi
                    cpu: 4
              endpoints:
              - name: predict
                port: 8080
                public: true
            """

            # Create service
            sql = f"""
            CREATE SERVICE {service_name}
            IN COMPUTE POOL {self.config.compute_pool}
            FROM SPECIFICATION $$
            {spec}
            $$
            """

            self.session.sql(sql).collect()

            # Get endpoint URL
            endpoint_url = f"https://{service_name}.snowflakecomputing.com/predict"

            return DeploymentResult(
                status="success",
                strategy=DeploymentStrategy.SPCS,
                target=self.config.target,
                deployment_name=self.config.deployment_name,
                service_name=service_name,
                endpoint_url=endpoint_url,
                start_time=start_time,
                end_time=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"SPCS deployment failed: {e}")
            return DeploymentResult(
                status="failed",
                strategy=DeploymentStrategy.SPCS,
                target=self.config.target,
                deployment_name=self.config.deployment_name,
                start_time=start_time,
                end_time=datetime.utcnow(),
                error=str(e),
            )

    def undeploy(self) -> bool:
        """Remove the SPCS service."""
        try:
            service_name = self.get_deployment_full_name()
            self.session.sql(f"DROP SERVICE IF EXISTS {service_name}").collect()
            return True
        except Exception as e:
            logger.error(f"Service removal failed: {e}")
            return False

    def validate(self) -> bool:
        """Validate the deployment configuration."""
        if not self.config.compute_pool:
            return False
        return bool(self.config.model_artifact_path and self.config.deployment_name)

    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            service_name = self.get_deployment_full_name()
            result = self.session.sql(
                f"SELECT SYSTEM$GET_SERVICE_STATUS('{service_name}')"
            ).collect()
            return {"status": result[0][0] if result else "unknown"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
