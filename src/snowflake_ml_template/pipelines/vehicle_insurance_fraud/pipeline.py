"""Vehicle insurance fraud detection pipeline.

Implements a complete fraud detection workflow including:
- Feature engineering for claim analysis and policy holder history
- Model training with LightGBM and class imbalance handling
- Deployment for batch and real-time scoring
- Monitoring for drift and performance metrics
"""

from typing import Any, Dict

from snowflake.snowpark.functions import avg, col, count, datediff
from snowflake.snowpark.functions import sum as sum_
from snowflake.snowpark.functions import when

from snowflake_ml_template.core.base.deployment import (
    DeploymentConfig,
    DeploymentStrategy,
    DeploymentTarget,
)
from snowflake_ml_template.core.base.training import (
    BaseModelConfig,
    MLFramework,
    TrainingConfig,
    TrainingStrategy,
)
from snowflake_ml_template.deployment.strategies.warehouse_udf import (
    WarehouseUDFStrategy,
)
from snowflake_ml_template.pipelines._base.template import PipelineTemplate
from snowflake_ml_template.registry import ModelRegistry, ModelStage
from snowflake_ml_template.training.frameworks.lightgbm_trainer import LightGBMTrainer
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)

# Type aliases
ModelResult = Dict[str, Any]


class VehicleInsuranceFraudPipeline(PipelineTemplate):
    """Vehicle insurance fraud detection using claim and vehicle data.

    Implements a complete fraud detection workflow including:
    - Feature engineering for claim analysis and policy holder history
    - Model training with LightGBM and class imbalance handling
    - Deployment for batch and real-time scoring
    - Monitoring for drift and performance metrics
    """

    def engineer_features(self) -> None:
        """Engineer features for vehicle insurance fraud detection."""
        self.logger.info("Engineering features for vehicle insurance fraud")

        # Register entities
        self.register_entity("CLAIM", ["CLAIM_ID"])
        self.register_entity("VEHICLE", ["VEHICLE_ID"])

        # Load claims data
        claims = self.session.table("RAW_DATA.INSURANCE_CLAIMS")

        # Claim features
        claim_features = claims.select(
            col("CLAIM_ID"),
            col("CLAIM_AMOUNT"),
            col("VEHICLE_AGE"),
            col("VEHICLE_VALUE"),
            col("CLAIM_TYPE"),
            when(col("CLAIM_AMOUNT") > col("VEHICLE_VALUE"), 1)
            .otherwise(0)
            .alias("AMOUNT_EXCEEDS_VALUE"),
            datediff("day", col("INCIDENT_DATE"), col("CLAIM_DATE")).alias(
                "DAYS_TO_CLAIM"
            ),
        )

        # Vehicle history features
        vehicle_features = claims.group_by("VEHICLE_ID").agg(
            [
                count("CLAIM_ID").alias("TOTAL_CLAIMS"),
                sum_("CLAIM_AMOUNT").alias("TOTAL_CLAIM_AMOUNT"),
                avg("CLAIM_AMOUNT").alias("AVG_CLAIM_AMOUNT"),
            ]
        )

        # Register feature views
        self.register_feature_view("claim_features", ["CLAIM"], claim_features)
        self.register_feature_view("vehicle_features", ["VEHICLE"], vehicle_features)

        self.logger.info("Feature engineering completed")

    def train_model(self) -> None:
        """Train the fraud detection model using LightGBM.

        Raises:
            RuntimeError: If model training fails
        """
        self.logger.info("Training fraud detection model")

        # Get training data from Feature Store
        training_data = self.session.table("FEATURES.VEHICLE_INSURANCE_TRAINING_DATA")

        # Configure model
        model_config = BaseModelConfig(
            framework=MLFramework.LIGHTGBM,
            model_type="classifier",
            hyperparameters={
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "seed": 42,
            },
            random_state=42,
        )

        # Configure training
        training_config = TrainingConfig(
            strategy=TrainingStrategy.SINGLE_NODE,
            model_config=model_config,
            training_database=self.config.database,
            training_schema="TRAINING",
            training_table="VEHICLE_INSURANCE_TRAINING_DATA",
            warehouse=self.config.warehouse,
            target_column="IS_FRAUD",
            validation_split=0.2,
            test_split=0.1,
        )

        # Initialize and train model
        trainer = LightGBMTrainer(training_config)
        result = trainer.train(training_data)

        if result.status == "success":
            self.model_version = "1.0.0"
            model_registry = ModelRegistry(
                session=self.session,
                database=self.config.database,
                schema="MODEL_REGISTRY",
            )
            model_registry.register_model(
                model_name="vehicle_insurance_fraud_detector",
                version=self.model_version,
                stage=ModelStage.DEV,
                artifact_path=result.model_artifact_path,
                framework="lightgbm",
                metrics=result.metrics,
                created_by="vehicle_insurance_fraud_pipeline",
            )
            self.logger.info(
                f"Model registered: vehicle_insurance_fraud_detector v{self.model_version}"
            )
        else:
            raise RuntimeError(f"Training failed: {result.error}")

    def deploy_model(self) -> None:
        """Deploy the trained model for inference.

        Raises:
            ValueError: If model version is not available or not found in registry
            RuntimeError: If deployment fails
        """
        self.logger.info("Deploying fraud detection model")

        if not self.model_version:
            raise ValueError("No model version available for deployment")

        # Get the model from registry
        model_registry = ModelRegistry(
            session=self.session, database=self.config.database, schema="MODEL_REGISTRY"
        )

        model_version = model_registry.get_version(
            model_name="vehicle_insurance_fraud_detector",
            version=str(self.model_version),
            stage=ModelStage.DEV,
        )

        if not model_version:
            raise ValueError(
                f"Model version {self.model_version} not found in registry"
            )

        # Configure deployment
        deployment_config = DeploymentConfig(
            strategy=DeploymentStrategy.WAREHOUSE_UDF,
            target=DeploymentTarget.BATCH,
            model_name=model_version.model_name,
            model_version=model_version.version,
            model_artifact_path=model_version.artifact_path,
            deployment_database=self.config.database,
            deployment_schema="DEPLOYMENT",
            deployment_name=f"vehicle_insurance_fraud_detector_v{self.model_version}",
            warehouse=self.config.warehouse,
        )

        # Deploy model
        try:
            deployer = WarehouseUDFStrategy(deployment_config)
            result = deployer.deploy()

            if result.status == "success":
                self.logger.info(
                    f"Model deployed successfully as {result.udf_name}",
                    extra={"udf_name": result.udf_name},
                )
            else:
                raise RuntimeError(f"Deployment failed: {result.error}")
        except Exception as e:
            self.logger.error(f"Model deployment failed: {str(e)}")
            raise RuntimeError(f"Failed to deploy model: {str(e)}")
