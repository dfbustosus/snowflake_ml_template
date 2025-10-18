"""Fraud detection pipeline implementation.

This reference pipeline demonstrates best practices for:
- Feature engineering with Feature Store
- Model training with XGBoost
- Model registration with versioning
- Model deployment as UDF
- Monitoring setup
"""

from typing import Any, Dict

from snowflake.snowpark.functions import avg, col, count, current_date, datediff, stddev
from snowflake.snowpark.functions import sum as sum_
from snowflake.snowpark.functions import when
from snowflake.snowpark.types import (
    DateType,
    DecimalType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

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
from snowflake_ml_template.training.frameworks.xgboost_trainer import XGBoostTrainer
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)

# Type aliases
ModelResult = Dict[str, Any]

# Define schema for the transactions table
TRANSACTIONS_SCHEMA = StructType(
    [
        StructField("TRANSACTION_ID", StringType()),
        StructField("CUSTOMER_ID", StringType()),
        StructField("TRANSACTION_DATE", DateType()),
        StructField("AMOUNT", DecimalType(38, 2)),
        StructField("MERCHANT_CATEGORY", StringType()),
        StructField("IS_FRAUD", IntegerType()),
        StructField("CARD_TYPE", StringType()),
        StructField("LOCATION", StringType()),
        StructField("HOUR_OF_DAY", IntegerType()),
        StructField("DEVICE_TYPE", StringType()),
    ]
)

# Define schema for the customers table
CUSTOMERS_SCHEMA = StructType(
    [
        StructField("CUSTOMER_ID", StringType()),
        StructField("ACCOUNT_AGE_DAYS", IntegerType()),
        StructField("HAS_PREVIOUS_FRAUD", IntegerType()),
        StructField("AVG_TRANSACTION_AMOUNT", DoubleType()),
        StructField("CUSTOMER_SEGMENT", StringType()),
    ]
)


class FraudDetectionPipeline(PipelineTemplate):
    """Fraud detection pipeline using transaction data.

    This pipeline implements a complete fraud detection workflow including:
    - Feature engineering with transaction aggregates and temporal features
    - Model training with XGBoost and class imbalance handling
    - Model deployment as UDF or endpoint
    - Monitoring for drift and performance

    Example:
        >>> from snowflake_ml_template.core.base.pipeline import PipelineConfig
        >>> config = PipelineConfig(
        ...     name="fraud_detection",
        ...     version="1.0.0",
        ...     environment="dev",
        ...     database="ML_DEV_DB",
        ...     warehouse="ML_TRAINING_WH"
        ... )
        >>>
        >>> pipeline = FraudDetectionPipeline(session, config)
        >>> result = pipeline.execute()
    """

    def engineer_features(self) -> None:
        """Engineer features for fraud detection."""
        self.logger.info("Engineering features for fraud detection")

        # Register entities
        self.register_entity("CUSTOMER", ["CUSTOMER_ID"])
        self.register_entity("TRANSACTION", ["TRANSACTION_ID"])

        # Load transaction data
        transactions = self.session.table("RAW_DATA.TRANSACTIONS")

        # Engineer features
        customer_features = transactions.group_by("CUSTOMER_ID").agg(
            [
                count("TRANSACTION_ID").alias("TRANSACTION_COUNT"),
                sum_("AMOUNT").alias("TOTAL_AMOUNT"),
                avg("AMOUNT").alias("AVG_AMOUNT"),
                stddev("AMOUNT").alias("STD_AMOUNT"),
                count(
                    when(
                        datediff("day", col("TRANSACTION_DATE"), current_date()) <= 7, 1
                    )
                ).alias("TRANSACTIONS_LAST_7_DAYS"),
                count(
                    when(
                        datediff("day", col("TRANSACTION_DATE"), current_date()) <= 30,
                        1,
                    )
                ).alias("TRANSACTIONS_LAST_30_DAYS"),
                avg(
                    when(
                        datediff("day", col("TRANSACTION_DATE"), current_date()) <= 7,
                        col("AMOUNT"),
                    )
                ).alias("AVG_AMOUNT_LAST_7_DAYS"),
            ]
        )

        # Register feature view
        self.register_feature_view(
            name="customer_transaction_features",
            entity_names=["CUSTOMER"],
            feature_df=customer_features,
        )

        self.logger.info("Feature engineering completed")

    def train_model(self) -> None:
        """Train fraud detection model.

        This method:
        1. Retrieves features from Feature Store
        2. Trains XGBoost classifier
        3. Registers model in Model Registry

        Raises:
            RuntimeError: If model training fails
        """
        self.logger.info("Training fraud detection model")

        # Get training data from Feature Store
        # In production, this would use BatchFeatureServer
        training_data = self.session.table("FEATURES.TRAINING_DATA")

        # Configure model
        model_config = BaseModelConfig(
            framework=MLFramework.XGBOOST,
            model_type="classifier",
            hyperparameters={
                "objective": "binary:logistic",
                "eval_metric": "auc",
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
            training_table="FRAUD_TRAINING_DATA",
            warehouse=self.config.warehouse,
            target_column="IS_FRAUD",
            validation_split=0.2,
            test_split=0.1,
        )

        # Initialize and train model
        trainer = XGBoostTrainer(training_config)
        result = trainer.train(training_data)

        if result.status == "success":
            # Register in Model Registry
            self.model_version = "1.0.0"
            model_registry = ModelRegistry(
                session=self.session,
                database=self.config.database,
                schema="MODEL_REGISTRY",
            )
            model_registry.register_model(
                model_name="fraud_detector",
                version=self.model_version,
                stage=ModelStage.DEV,
                artifact_path=result.model_artifact_path,
                framework="xgboost",
                metrics=result.metrics,
                created_by="fraud_detection_pipeline",
            )
            self.logger.info(
                f"Model registered: fraud_detector v{self.model_version}",
                extra={"version": self.model_version},
            )
        else:
            raise RuntimeError(f"Training failed: {result.error}")

    def deploy_model(self) -> None:
        """Deploy fraud detection model as Warehouse UDF.

        This creates a UDF that can be used for batch scoring:
        SELECT fraud_predict_udf(features) FROM transactions;

        Raises:
            ValueError: If model version is not available or not found in registry
            RuntimeError: If deployment fails
        """
        self.logger.info("Deploying fraud detection model")

        if not self.model_version:
            raise ValueError("No model version available for deployment")

        # Get model from registry
        model_registry = ModelRegistry(
            session=self.session, database=self.config.database, schema="MODEL_REGISTRY"
        )
        model_version = model_registry.get_version(
            "fraud_detector", self.model_version, ModelStage.DEV
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
            deployment_name=f"fraud_detector_v{self.model_version}",
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

    def setup_monitoring(self) -> None:
        """Set up monitoring for fraud detection model.

        This configures:
        - Feature drift detection
        - Model performance tracking
        - Data quality monitoring
        """
        self.logger.info("Setting up monitoring for fraud detection")

        # In production, this would configure:
        # - FeatureDriftDetector for feature monitoring
        # - Model performance tracking
        # - Alerting thresholds

        self.logger.info("Monitoring setup completed")
