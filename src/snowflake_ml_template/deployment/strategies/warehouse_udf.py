"""Warehouse UDF deployment strategy for batch inference."""

from datetime import datetime
from typing import Any, Dict, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.deployment import (
    BaseDeploymentStrategy,
    DeploymentConfig,
    DeploymentResult,
    DeploymentStrategy,
)
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class WarehouseUDFStrategy(BaseDeploymentStrategy):
    """Deploy model as Warehouse UDF for batch inference.

    Optimal for:
    - Batch scoring (seconds to minutes latency)
    - Large-scale batch processing
    - Cost-effective inference
    - Anaconda package dependencies

    Example:
        >>> config = DeploymentConfig(
        ...     strategy=DeploymentStrategy.WAREHOUSE_UDF,
        ...     target=DeploymentTarget.BATCH,
        ...     model_name="fraud_detector",
        ...     model_version="1.0.0",
        ...     model_artifact_path="@ML_MODELS_STAGE/fraud_v1.joblib",
        ...     deployment_database="ML_PROD_DB",
        ...     deployment_schema="MODELS",
        ...     deployment_name="fraud_predict_udf",
        ...     warehouse="INFERENCE_WH"
        ... )
        >>> strategy = WarehouseUDFStrategy(config)
        >>> result = strategy.deploy()
    """

    def __init__(self, config: DeploymentConfig) -> None:
        """Initialize the Warehouse UDF strategy.

        Args:
            config: Deployment configuration
        """
        super().__init__(config)
        self._session: Optional[Session] = None

    def set_session(self, session: Session) -> None:
        """Set the Snowflake session.

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

    def deploy(self, **kwargs: Any) -> DeploymentResult:
        """Deploy model as Warehouse UDF."""
        start_time = datetime.utcnow()

        try:
            udf_name = self.get_deployment_full_name()

            # Upload model to stage if it's a local file
            stage_path = self.config.model_artifact_path
            if not stage_path.startswith("@"):
                # Local file - upload to stage
                stage_name = f"{self.config.deployment_database}.{self.config.deployment_schema}.ML_MODELS_STAGE"
                import os

                model_filename = os.path.basename(stage_path)

                # Upload file to stage
                self.session.file.put(
                    stage_path, f"@{stage_name}", auto_compress=False, overwrite=True
                )
                stage_path = f"@{stage_name}/{model_filename}"
                logger.info(f"Uploaded model to {stage_path}")

            # Create UDF
            sql = f"""
            CREATE OR REPLACE FUNCTION {udf_name}(features VARIANT)
            RETURNS VARIANT
            LANGUAGE PYTHON
            RUNTIME_VERSION = '3.10'
            PACKAGES = ('scikit-learn', 'xgboost', 'lightgbm', 'joblib', 'pandas', 'numpy')
            IMPORTS = ('{stage_path}')
            HANDLER = 'predict'
            AS
            $$
import joblib
import sys
import os
import pandas as pd
import time
import uuid
from datetime import datetime

# Global model cache to avoid reloading on every call
_model = None
_feature_names = None

def predict(features):
    global _model, _feature_names
    start_time = time.time()
    # Load model once and cache it
    if _model is None:
        import_dir = sys._xoptions.get('snowflake_import_directory')
        model_files = [f for f in os.listdir(import_dir) if f.endswith('.joblib')]
        if not model_files:
            raise ValueError("No model file found in imports")
        model_path = os.path.join(import_dir, model_files[0])
        _model = joblib.load(model_path)
        # Get feature names from model
        if hasattr(_model, 'feature_name_'):
            _feature_names = _model.feature_name_
        elif hasattr(_model, 'feature_names_in_'):
            _feature_names = list(_model.feature_names_in_)
        else:
            _feature_names = None
    # Convert dict to DataFrame with single row
    df = pd.DataFrame([features])
    # Reorder columns to match training order if we have feature names
    if _feature_names is not None:
        # Only use features that exist in both the input and training
        available_features = [f for f in _feature_names if f in df.columns]
        df = df[available_features]
    # Make prediction
    prediction = _model.predict(df)
    pred_value = float(prediction[0])
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    # Log inference (Note: UDFs cannot directly INSERT, so we return metadata)
    # For production, use external logging or Snowpipe Streaming
    # This is a simplified version that returns the prediction
    return pred_value
$$
            """

            self.session.sql(sql).collect()

            return DeploymentResult(
                status="success",
                strategy=DeploymentStrategy.WAREHOUSE_UDF,
                target=self.config.target,
                deployment_name=self.config.deployment_name,
                udf_name=udf_name,
                start_time=start_time,
                end_time=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"UDF deployment failed: {e}")
            return DeploymentResult(
                status="failed",
                strategy=DeploymentStrategy.WAREHOUSE_UDF,
                target=self.config.target,
                deployment_name=self.config.deployment_name,
                start_time=start_time,
                end_time=datetime.utcnow(),
                error=str(e),
            )

    def undeploy(self) -> bool:
        """Remove the UDF."""
        try:
            udf_name = self.get_deployment_full_name()
            self.session.sql(f"DROP FUNCTION IF EXISTS {udf_name}").collect()
            return True
        except Exception as e:
            logger.error(f"UDF removal failed: {e}")
            return False

    def validate(self) -> bool:
        """Validate the deployment configuration."""
        return bool(self.config.model_artifact_path and self.config.deployment_name)

    def health_check(self) -> Dict[str, Any]:
        """Check if UDF is accessible."""
        try:
            result = self.session.sql(
                f"SHOW FUNCTIONS LIKE '{self.config.deployment_name}'"
            ).collect()
            return {"status": "healthy" if result else "unavailable"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
