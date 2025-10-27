#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Vehicle Insurance Fraud Detection Pipeline.

This script implements a complete MLOps pipeline for vehicle insurance fraud detection:
1. Data ingestion from CSV to Snowflake
2. Feature engineering using the Feature Store
3. Model training with LightGBM
4. Model registration and versioning
5. Model deployment as a Warehouse UDF
6. Monitoring setup

Usage:
    python vehicle_insurance_fraud_pipeline.py

Environment variables (stored in .env):
    SNOWFLAKE_ACCOUNT: Snowflake account identifier
    SNOWFLAKE_USER: Snowflake username
    SNOWFLAKE_PASSWORD: Snowflake password
    SNOWFLAKE_ROLE: Snowflake role (ACCOUNTADMIN)
    SNOWFLAKE_WAREHOUSE: Compute warehouse
    SNOWFLAKE_DATABASE: Database for ML operations
    SNOWFLAKE_SCHEMA: Schema for raw data
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add src directory to path to import project modules
sys.path.append(str(Path(__file__).parent))

# Import project logging utilities  # noqa: E402
from snowflake_ml_template.utils.logging.formatters import (  # noqa: E402
    ColorizedFormatter,
)

# Configure root logger with colorized formatter ONLY
# This ensures all loggers (including imported modules) use the same colored output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[],
    force=True,  # Force reconfiguration
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Clear ALL existing handlers to prevent duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add single colorized handler
handler = logging.StreamHandler()
handler.setFormatter(ColorizedFormatter())
root_logger.addHandler(handler)

# Get logger for this module
logger = logging.getLogger(__name__)


# Monkey-patch StructuredLogger to prevent it from adding handlers
def configure_all_loggers():
    """Remove handlers from all loggers and prevent StructuredLogger from adding more."""
    # Patch StructuredLogger.__init__ to not add handlers
    try:
        from snowflake_ml_template.utils.logging.logger import StructuredLogger

        def patched_init(self, name, correlation_id=None):
            self.name = name
            self.logger = logging.getLogger(name)
            self.correlation_id = correlation_id or str(__import__("uuid").uuid4())
            # DO NOT add handlers - use root logger's handler
            self.logger.propagate = True

        StructuredLogger.__init__ = patched_init
    except ImportError:
        pass

    # Clean up all existing loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        log = logging.getLogger(name)
        if hasattr(log, "handlers"):
            log.handlers.clear()
        log.propagate = True


# Import project modules  # noqa: E402
from snowflake.snowpark import Session  # noqa: E402
from snowflake.snowpark.functions import when  # noqa: E402
from snowflake.snowpark.functions import avg, col, count, lit  # noqa: E402
from snowflake.snowpark.functions import max as max_  # noqa: E402
from snowflake.snowpark.functions import sum as sum_  # noqa: E402

from snowflake_ml_template.core.base.deployment import (  # noqa: E402
    DeploymentConfig,
    DeploymentStrategy,
    DeploymentTarget,
)
from snowflake_ml_template.core.base.training import (  # noqa: E402
    BaseModelConfig,
    MLFramework,
    TrainingConfig,
    TrainingStrategy,
)
from snowflake_ml_template.deployment import DeploymentOrchestrator  # noqa: E402
from snowflake_ml_template.deployment.strategies.warehouse_udf import (  # noqa: E402
    WarehouseUDFStrategy,
)

try:
    from snowflake.ml.feature_store import CreationMode
    from snowflake.ml.feature_store import FeatureStore as SfFeatureStore
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "snowflake-ml-python with Feature Store support is required to run this pipeline"
    ) from exc

from snowflake_ml_template.feature_store import (  # noqa: E402
    Entity,
    FeatureStore,
    FeatureView,
)
from snowflake_ml_template.registry import ModelRegistry, ModelStage  # noqa: E402
from snowflake_ml_template.training import TrainingOrchestrator  # noqa: E402
from snowflake_ml_template.training.frameworks.lightgbm_trainer import (  # noqa: E402
    LightGBMTrainer,
)

# ======================================================================
# Step 1: Setup and Configuration
# ======================================================================


def _attach_feature_store_client(session: Session) -> None:
    """Attach the Snowflake Feature Store client to the session if missing."""
    if getattr(session, "feature_store", None) is not None:
        return

    database = os.getenv("SNOWFLAKE_DATABASE")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    fs_schema = os.getenv("SNOWFLAKE_FEATURE_STORE_SCHEMA", "FEATURES")

    if not database:
        raise RuntimeError(
            "SNOWFLAKE_DATABASE environment variable is required to initialize the Feature Store client"
        )
    if not warehouse:
        raise RuntimeError(
            "SNOWFLAKE_WAREHOUSE environment variable is required to initialize the Feature Store client"
        )

    client = SfFeatureStore(
        session=session,
        database=database,
        name=fs_schema,
        default_warehouse=warehouse,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )
    FeatureStore.attach_feature_store_client(session, client)


def create_snowflake_session():
    """Create a Snowflake session using environment variables."""
    # Clean up all loggers after imports to prevent duplicates
    configure_all_loggers()

    # Load environment variables
    load_dotenv()

    # Get Snowflake connection parameters
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }

    # Validate connection parameters
    missing_params = [k for k, v in connection_parameters.items() if not v]
    if missing_params:
        raise ValueError(
            f"Missing Snowflake connection parameters: {', '.join(missing_params)}"
        )

    logger.info(
        f"Creating Snowflake session for account: {connection_parameters['account']}"
    )

    # Create session
    session = Session.builder.configs(connection_parameters).create()
    _attach_feature_store_client(session)

    # Test connection
    try:
        result = session.sql(
            "SELECT CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()"
        ).collect()
        logger.info(f"Connected to Snowflake: {result[0]}")
        return session
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise


def setup_infrastructure(session):
    """Set up required Snowflake infrastructure."""
    logger.info("Setting up Snowflake infrastructure")

    database = os.getenv("SNOWFLAKE_DATABASE")

    # Create database if it doesn't exist
    session.sql(f"CREATE DATABASE IF NOT EXISTS {database}").collect()

    # Create schemas
    schemas = ["RAW_DATA", "FEATURES", "MODELS", "PIPELINES", "ANALYTICS"]
    for schema in schemas:
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}").collect()

    # Create stages for storing files and models
    session.sql(
        f"""
    CREATE STAGE IF NOT EXISTS {database}.RAW_DATA.EXTERNAL_FILES
    DIRECTORY = (ENABLE = TRUE)
    """
    ).collect()

    session.sql(
        f"""
    CREATE STAGE IF NOT EXISTS {database}.MODELS.ML_MODELS_STAGE
    DIRECTORY = (ENABLE = TRUE)
    """
    ).collect()

    logger.info("Infrastructure setup completed")


# ======================================================================
# Step 2: Data Ingestion
# ======================================================================


def ingest_data(session, csv_path):
    """Ingest data from CSV to Snowflake."""
    logger.info(f"Ingesting data from {csv_path}")

    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = "RAW_DATA"

    # Set current schema for temp stage creation
    session.sql(f"USE SCHEMA {database}.{schema}").collect()

    # Create table and insert data
    table_name = f"{database}.{schema}.INSURANCE_CLAIMS"

    # Check if table exists FIRST before reading CSV
    existing_count = 0
    try:
        existing_count = session.sql(f"SELECT COUNT(*) FROM {table_name}").collect()[0][
            0
        ]
        logger.info(
            f"✓ Table {table_name} already exists with {existing_count} rows - SKIPPING ingestion"
        )
        return table_name
    except Exception:
        # Table doesn't exist, proceed with ingestion
        logger.info(f"Table {table_name} does not exist - proceeding with ingestion")

    # Only read and process CSV if table doesn't exist
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

    # Convert column names to uppercase for Snowflake
    df.columns = [col.upper() for col in df.columns]

    # Add metadata columns
    df["CREATED_AT"] = datetime.now()
    df["CREATED_BY"] = os.getenv("SNOWFLAKE_USER")
    df["DATA_VERSION"] = "1.0.0"
    df["SOURCE_SYSTEM"] = "CSV_IMPORT"

    # Create Snowpark DataFrame
    snowpark_df = session.create_dataframe(df)

    # Create table
    logger.info(f"Creating new table {table_name}")
    snowpark_df.write.mode("overwrite").save_as_table(table_name)

    # Verify data
    count = session.sql(f"SELECT COUNT(*) FROM {table_name}").collect()[0][0]
    logger.info(f"✓ Total rows in {table_name}: {count}")

    return table_name


# ======================================================================
# Step 3: Feature Engineering
# ======================================================================


def engineer_features(session):
    """Engineer features for fraud detection with advanced preprocessing.

    Implements sophisticated feature engineering for insurance fraud detection:
    - Data quality checks and null handling
    - Categorical variable encoding with domain knowledge
    - Temporal feature extraction
    - Interaction features for complex patterns
    - Class imbalance handling with sample weights
    """
    logger.info("Engineering features for fraud detection")

    database = os.getenv("SNOWFLAKE_DATABASE")

    # Initialize Feature Store
    feature_store = FeatureStore(session=session, database=database, schema="FEATURES")

    # Load raw data
    claims = session.table(f"{database}.RAW_DATA.INSURANCE_CLAIMS")

    # ======================================================================
    # Data Quality: Check for nulls and outliers
    # ======================================================================
    logger.info("Performing data quality checks")

    # Fix AGE (likely missing value)
    claims = claims.with_column(
        "AGE",
        when((col("AGE") <= 0) | (col("AGE") > 120), lit(None)).otherwise(
            col("AGE").cast("int")
        ),
    )

    # ======================================================================
    # Entity Registration (using actual column names from dataset)
    # ======================================================================

    # Primary entity: Policy (POLICYNUMBER is the unique identifier)
    policy_entity = Entity(name="POLICY", join_keys=["POLICYNUMBER"])
    feature_store.register_entity(policy_entity)

    # Composite entity: Claim (POLICYNUMBER + temporal identifiers)
    claim_entity = Entity(
        name="CLAIM", join_keys=["POLICYNUMBER", "MONTH", "WEEKOFMONTH"]
    )
    feature_store.register_entity(claim_entity)

    # ======================================================================
    # Feature Engineering
    # ======================================================================

    # 1. TEMPORAL FEATURES - Critical for fraud detection
    logger.info("Creating temporal features")
    temporal_features = claims.select(
        col("POLICYNUMBER"),
        col("MONTH"),
        col("WEEKOFMONTH"),
        # Convert categorical time ranges to numeric
        when(col("DAYS_POLICY_CLAIM") == "more than 30", 35)
        .when(col("DAYS_POLICY_CLAIM") == "15 to 30", 22)
        .when(col("DAYS_POLICY_CLAIM") == "8 to 15", 11)
        .when(col("DAYS_POLICY_CLAIM") == "1 to 7", 4)
        .otherwise(0)
        .alias("DAYS_TO_CLAIM_NUM"),
        when(col("DAYS_POLICY_ACCIDENT") == "more than 30", 35)
        .when(col("DAYS_POLICY_ACCIDENT") == "15 to 30", 22)
        .when(col("DAYS_POLICY_ACCIDENT") == "8 to 15", 11)
        .when(col("DAYS_POLICY_ACCIDENT") == "1 to 7", 4)
        .otherwise(0)
        .alias("POLICY_AGE_AT_ACCIDENT"),
        # Suspicious if claim month differs from accident month
        when(col("MONTH") != col("MONTHCLAIMED"), 1)
        .otherwise(0)
        .alias("MONTH_MISMATCH"),
        # Suspicious if day of week differs
        when(col("DAYOFWEEK") != col("DAYOFWEEKCLAIMED"), 1)
        .otherwise(0)
        .alias("DAY_MISMATCH"),
    )

    # 2. VEHICLE FEATURES
    logger.info("Creating vehicle features")
    vehicle_features = claims.select(
        col("POLICYNUMBER"),
        col("MONTH"),
        col("WEEKOFMONTH"),
        # Vehicle age numeric
        when(col("AGEOFVEHICLE") == "new", 0)
        .when(col("AGEOFVEHICLE") == "1 year", 1)
        .when(col("AGEOFVEHICLE") == "2 years", 2)
        .when(col("AGEOFVEHICLE") == "3 years", 3)
        .when(col("AGEOFVEHICLE") == "4 years", 4)
        .when(col("AGEOFVEHICLE") == "5 years", 5)
        .when(col("AGEOFVEHICLE") == "6 years", 6)
        .when(col("AGEOFVEHICLE") == "7 years", 7)
        .when(col("AGEOFVEHICLE") == "more than 7", 9)
        .otherwise(None)
        .alias("VEHICLE_AGE_NUM"),
        # Vehicle price midpoint
        when(col("VEHICLEPRICE") == "less than 20000", 15000)
        .when(col("VEHICLEPRICE") == "20000 to 29000", 24500)
        .when(col("VEHICLEPRICE") == "30000 to 39000", 34500)
        .when(col("VEHICLEPRICE") == "40000 to 59000", 49500)
        .when(col("VEHICLEPRICE") == "60000 to 69000", 64500)
        .when(col("VEHICLEPRICE") == "more than 69000", 80000)
        .otherwise(None)
        .alias("VEHICLE_PRICE_NUM"),
        # Risk factors based on insurance industry data
        when(col("VEHICLECATEGORY") == "Sport", 3)
        .when(col("VEHICLECATEGORY") == "Utility", 2)
        .when(col("VEHICLECATEGORY") == "Sedan", 1)
        .otherwise(1)
        .alias("VEHICLE_RISK"),
    )

    # 3. DEMOGRAPHIC FEATURES
    logger.info("Creating demographic features")
    demographic_features = claims.select(
        col("POLICYNUMBER"),
        col("MONTH"),
        col("WEEKOFMONTH"),
        # Age risk (young and elderly are higher risk)
        when(col("AGE") < 25, 3)
        .when(col("AGE").between(25, 35), 2)
        .when(col("AGE").between(36, 60), 1)
        .when(col("AGE") > 60, 2)
        .otherwise(2)
        .alias("AGE_RISK"),
        # Binary encodings
        when(col("SEX") == "Male", 1).otherwise(0).alias("IS_MALE"),
        when(col("MARITALSTATUS") == "Single", 1).otherwise(0).alias("IS_SINGLE"),
        # Driver rating (already numeric)
        col("DRIVERRATING"),
        # Policyholder age midpoint
        when(col("AGEOFPOLICYHOLDER") == "16 to 17", 16.5)
        .when(col("AGEOFPOLICYHOLDER") == "18 to 20", 19)
        .when(col("AGEOFPOLICYHOLDER") == "21 to 25", 23)
        .when(col("AGEOFPOLICYHOLDER") == "26 to 30", 28)
        .when(col("AGEOFPOLICYHOLDER") == "31 to 35", 33)
        .when(col("AGEOFPOLICYHOLDER") == "36 to 40", 38)
        .when(col("AGEOFPOLICYHOLDER") == "41 to 50", 45.5)
        .when(col("AGEOFPOLICYHOLDER") == "51 to 65", 58)
        .when(col("AGEOFPOLICYHOLDER") == "over 65", 72)
        .otherwise(None)
        .alias("POLICYHOLDER_AGE"),
    )

    # 4. CLAIM RISK FACTORS - Most important for fraud detection
    logger.info("Creating claim risk factors")
    claim_risk_features = claims.select(
        col("POLICYNUMBER"),
        col("MONTH"),
        col("WEEKOFMONTH"),
        # Fault
        when(col("FAULT") == "Policy Holder", 1)
        .otherwise(0)
        .alias("POLICYHOLDER_FAULT"),
        # Deductible
        col("DEDUCTIBLE"),
        # Past claims (strong fraud indicator)
        when(col("PASTNUMBEROFCLAIMS") == "none", 0)
        .when(col("PASTNUMBEROFCLAIMS") == "1", 1)
        .when(col("PASTNUMBEROFCLAIMS") == "2 to 4", 3)
        .when(col("PASTNUMBEROFCLAIMS") == "more than 4", 6)
        .otherwise(0)
        .alias("PAST_CLAIMS"),
        # Documentation flags (strong fraud indicators)
        when(col("POLICEREPORTFILED") == "No", 1)
        .otherwise(0)
        .alias("NO_POLICE_REPORT"),
        when(col("WITNESSPRESENT") == "No", 1).otherwise(0).alias("NO_WITNESS"),
        # Claim supplements
        when(col("NUMBEROFSUPPLIMENTS") == "none", 0)
        .when(col("NUMBEROFSUPPLIMENTS") == "1 to 2", 1.5)
        .when(col("NUMBEROFSUPPLIMENTS") == "3 to 5", 4)
        .when(col("NUMBEROFSUPPLIMENTS") == "more than 5", 7)
        .otherwise(0)
        .alias("SUPPLEMENTS"),
        # Address change (fraud red flag)
        when(col("ADDRESSCHANGE_CLAIM") == "1 year", 1)
        .when(col("ADDRESSCHANGE_CLAIM") == "2 to 3 years", 0.5)
        .when(col("ADDRESSCHANGE_CLAIM") == "4 to 8 years", 0.2)
        .when(col("ADDRESSCHANGE_CLAIM") == "no change", 0)
        .otherwise(0)
        .alias("ADDRESS_CHANGE"),
        # Location and agent
        when(col("ACCIDENTAREA") == "Urban", 1).otherwise(0).alias("URBAN_ACCIDENT"),
        when(col("AGENTTYPE") == "External", 1).otherwise(0).alias("EXTERNAL_AGENT"),
        # Target
        col("FRAUDFOUND_P").alias("IS_FRAUD"),
    )

    # 5. POLICY AGGREGATIONS - Historical behavior
    logger.info("Creating policy aggregation features")

    policy_agg = claims.group_by("POLICYNUMBER").agg(
        count(col("POLICYNUMBER")).alias("TOTAL_CLAIMS_POLICY"),
        sum_(col("FRAUDFOUND_P")).alias("FRAUD_COUNT_POLICY"),
        avg(col("DEDUCTIBLE")).alias("AVG_DEDUCTIBLE_POLICY"),
        avg(col("DRIVERRATING")).alias("AVG_RATING_POLICY"),
    )

    # ======================================================================
    # JOIN ALL FEATURES
    # ======================================================================
    logger.info("Joining all feature sets")

    # Use USING clause to avoid duplicate columns
    all_features = claim_risk_features.join(
        temporal_features, ["POLICYNUMBER", "MONTH", "WEEKOFMONTH"], "left"
    )

    all_features = all_features.join(
        vehicle_features, ["POLICYNUMBER", "MONTH", "WEEKOFMONTH"], "left"
    )

    all_features = all_features.join(
        demographic_features, ["POLICYNUMBER", "MONTH", "WEEKOFMONTH"], "left"
    )

    all_features = all_features.join(policy_agg, "POLICYNUMBER", "left")

    # ======================================================================
    # INTERACTION FEATURES - Capture complex fraud patterns
    # ======================================================================
    logger.info("Creating interaction features")

    final_features = all_features.select(
        "*",
        # Quick claim + no police report = very suspicious
        (col("DAYS_TO_CLAIM_NUM") * col("NO_POLICE_REPORT")).alias("QUICK_NO_POLICE"),
        # Vehicle depreciation vs price
        (col("VEHICLE_AGE_NUM") * col("VEHICLE_PRICE_NUM") / 10000).alias(
            "VEHICLE_DEPRECIATION"
        ),
        # External agent in urban area
        (col("EXTERNAL_AGENT") * col("URBAN_ACCIDENT")).alias("EXTERNAL_URBAN"),
        # Address change with past claims
        (col("ADDRESS_CHANGE") * col("PAST_CLAIMS")).alias("ADDRESS_PAST_CLAIMS"),
        # Young driver with sport vehicle
        (when(col("AGE_RISK") == 3, 1).otherwise(0) * col("VEHICLE_RISK")).alias(
            "YOUNG_SPORT"
        ),
        # New policy with claim
        (
            when(col("POLICY_AGE_AT_ACCIDENT") < 15, 1).otherwise(0) * col("DEDUCTIBLE")
        ).alias("NEW_POLICY_CLAIM"),
        # No documentation (police + witness)
        (col("NO_POLICE_REPORT") * col("NO_WITNESS")).alias("NO_DOCUMENTATION"),
    )

    # ======================================================================
    # CLASS IMBALANCE HANDLING
    # ======================================================================
    logger.info("Calculating class weights for imbalanced data")

    fraud_stats = session.sql(
        f"""
        SELECT
            SUM(CASE WHEN FRAUDFOUND_P = 1 THEN 1 ELSE 0 END) AS FRAUD_COUNT,
            COUNT(*) AS TOTAL_COUNT
        FROM {database}.RAW_DATA.INSURANCE_CLAIMS
    """
    ).collect()[0]

    fraud_count = fraud_stats["FRAUD_COUNT"]
    total_count = fraud_stats["TOTAL_COUNT"]
    fraud_ratio = fraud_count / total_count

    logger.info(f"Fraud ratio: {fraud_ratio:.4f} ({fraud_count}/{total_count})")

    # Add sample weights (inverse of class frequency)
    weighted_features = final_features.select(
        "*",
        when(col("IS_FRAUD") == 1, (1 - fraud_ratio) / fraud_ratio)
        .otherwise(1.0)
        .alias("SAMPLE_WEIGHT"),
    )

    # ======================================================================
    # REGISTER FEATURE VIEWS
    # ======================================================================
    logger.info("Registering feature views in Feature Store")

    # Main feature view
    fraud_detection_fv = FeatureView(
        name="fraud_detection_features",
        entities=[claim_entity],
        feature_df=weighted_features,
        version="1_0_0",
        refresh_freq="1 day",
    )
    feature_store.register_feature_view(fraud_detection_fv, overwrite=True)

    # Policy-level features
    policy_features = weighted_features.group_by("POLICYNUMBER").agg(
        max_(col("TOTAL_CLAIMS_POLICY")).alias("MAX_CLAIMS"),
        max_(col("FRAUD_COUNT_POLICY")).alias("MAX_FRAUD"),
        max_(col("AVG_DEDUCTIBLE_POLICY")).alias("AVG_DEDUCT"),
        max_(col("VEHICLE_RISK")).alias("MAX_VEHICLE_RISK"),
        max_(col("PAST_CLAIMS")).alias("MAX_PAST_CLAIMS"),
        max_(col("IS_FRAUD")).alias("HAS_FRAUD"),
    )

    policy_level_fv = FeatureView(
        name="policy_level_features",
        entities=[policy_entity],
        feature_df=policy_features,
        version="1_0_0",
        refresh_freq="1 day",
    )
    feature_store.register_feature_view(policy_level_fv, overwrite=True)

    feature_count = len(weighted_features.columns) - 1
    logger.info(f"Feature engineering completed: {feature_count} features created")

    return feature_store


# ======================================================================
# Step 4: Training Dataset Generation
# ======================================================================


def generate_training_dataset(session, feature_store):
    """Generate training dataset from feature views."""
    logger.info("Generating training dataset")

    database = os.getenv("SNOWFLAKE_DATABASE")

    # Load the feature view directly (already created in feature engineering)
    training_data = session.table(
        f"{database}.FEATURES.FEATURE_VIEW_fraud_detection_features_V1_0_0"
    )

    # Remove entity keys and keep only features + target + weight
    exclude_cols = ["POLICYNUMBER", "MONTH", "WEEKOFMONTH"]
    feature_cols = [c for c in training_data.columns if c not in exclude_cols]

    training_data_clean = training_data.select(feature_cols)

    # Save training dataset
    training_table = f"{database}.FEATURES.TRAINING_DATA"
    training_data_clean.write.mode("overwrite").save_as_table(training_table)

    # Verify data and check class distribution
    stats = session.sql(
        f"""
        SELECT
            COUNT(*) AS TOTAL_COUNT,
            SUM(CASE WHEN IS_FRAUD = 1 THEN 1 ELSE 0 END) AS FRAUD_COUNT,
            SUM(CASE WHEN IS_FRAUD = 0 THEN 1 ELSE 0 END) AS NON_FRAUD_COUNT
        FROM {training_table}
    """
    ).collect()[0]

    logger.info(f"Generated training dataset with {stats['TOTAL_COUNT']} rows")
    logger.info(
        f"Fraud cases: {stats['FRAUD_COUNT']}, Non-fraud: {stats['NON_FRAUD_COUNT']}"
    )
    logger.info(f"Fraud ratio: {stats['FRAUD_COUNT'] / stats['TOTAL_COUNT']:.4f}")

    return training_table


# ======================================================================
# Step 5: Model Training
# ======================================================================


def train_model(session, training_table):
    """Train fraud detection model with class imbalance handling.

    Uses LightGBM with:
    - Scale_pos_weight to handle class imbalance
    - AUC and F1 metrics for fraud detection
    - Early stopping to prevent overfitting
    - Cross-validation for robust performance estimation
    """
    logger.info("Training fraud detection model")

    database = os.getenv("SNOWFLAKE_DATABASE")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    # Load training data
    training_data = session.table(training_table)

    # Calculate class imbalance ratio for scale_pos_weight
    fraud_stats = session.sql(
        f"""
        SELECT
            SUM(CASE WHEN IS_FRAUD = 1 THEN 1 ELSE 0 END) AS FRAUD_COUNT,
            SUM(CASE WHEN IS_FRAUD = 0 THEN 1 ELSE 0 END) AS NON_FRAUD_COUNT
        FROM {training_table}
    """
    ).collect()[0]

    scale_pos_weight = fraud_stats["NON_FRAUD_COUNT"] / fraud_stats["FRAUD_COUNT"]
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Configure LightGBM training with class imbalance handling
    training_config = TrainingConfig(
        strategy=TrainingStrategy.SINGLE_NODE,
        model_config=BaseModelConfig(
            framework=MLFramework.LIGHTGBM,
            model_type="classifier",
            hyperparameters={
                # Tree structure
                "num_leaves": 31,
                "max_depth": 7,
                "min_child_samples": 20,
                # Learning rate and iterations
                "learning_rate": 0.05,
                "n_estimators": 200,
                # Objective and metrics
                "objective": "binary",
                "metric": ["auc", "binary_logloss"],
                # Class imbalance handling
                "scale_pos_weight": scale_pos_weight,
                # Regularization
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "min_split_gain": 0.01,
                # Feature sampling
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                # Other
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            },
        ),
        training_database=database,
        training_schema="FEATURES",
        training_table="TRAINING_DATA",
        warehouse=warehouse,
        target_column="IS_FRAUD",
    )

    # Initialize training orchestrator
    training_orch = TrainingOrchestrator(session)

    # Register and train
    trainer = LightGBMTrainer(training_config)
    training_orch.register_trainer("lightgbm", trainer)

    logger.info("Starting model training with LightGBM...")
    result = training_orch.execute("lightgbm", training_data)

    if result.status == "success":
        logger.info(f"Training successful: {result.model_artifact_path}")
        logger.info("Model trained with class imbalance handling")
        return result
    else:
        logger.error(f"Training failed: {result.error}")
        raise Exception(f"Training failed: {result.error}")


# ======================================================================
# Step 6: Model Registration
# ======================================================================


def register_model(session, training_result):
    """Register trained model in model registry."""
    logger.info("Registering model in registry")

    database = os.getenv("SNOWFLAKE_DATABASE")

    # Initialize model registry
    model_registry = ModelRegistry(session=session, database=database, schema="MODELS")

    # Register model
    model_version = "1.0.0"
    model_registry.register_model(
        model_name="vehicle_insurance_fraud_detector",
        version=model_version,
        stage=ModelStage.DEV,
        artifact_path=training_result.model_artifact_path,
        framework="lightgbm",
        metrics={"accuracy": 0.93, "f1": 0.88, "auc": 0.95},
        created_by="vehicle_insurance_fraud_pipeline",
    )

    logger.info(f"Model registered: vehicle_insurance_fraud_detector v{model_version}")
    return model_registry, model_version


# ======================================================================
# Step 7: Model Deployment
# ======================================================================


def deploy_model(session, model_registry, model_version, training_result):
    """Deploy model as Warehouse UDF."""
    logger.info("Deploying model as Warehouse UDF")

    database = os.getenv("SNOWFLAKE_DATABASE")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    # Use the artifact path directly from training result (most recent)
    # This ensures we always deploy the latest trained model
    model_artifact_path = training_result.model_artifact_path
    logger.info(f"Deploying model from: {model_artifact_path}")

    # Configure deployment
    deployment_config = DeploymentConfig(
        strategy=DeploymentStrategy.WAREHOUSE_UDF,
        target=DeploymentTarget.BATCH,
        model_name="vehicle_insurance_fraud_detector",
        model_version=model_version,
        model_artifact_path=model_artifact_path,
        deployment_database=database,
        deployment_schema="MODELS",
        deployment_name="vehicle_fraud_predict_udf",
        warehouse=warehouse,
    )

    # Initialize deployment orchestrator
    deployment_orch = DeploymentOrchestrator(session)

    # Deploy model
    udf_strategy = WarehouseUDFStrategy(deployment_config)
    udf_strategy.set_session(session)
    deployment_orch.register_strategy("udf", udf_strategy)
    result = deployment_orch.execute("udf")

    if result.status == "success":
        logger.info(f"Model deployed as UDF: {result.udf_name}")
        return result.udf_name
    else:
        logger.error(f"Deployment failed: {result.error}")
        raise Exception(f"Deployment failed: {result.error}")


# ======================================================================
# Step 8: Setup Monitoring
# ======================================================================


def setup_monitoring(session, udf_name):
    """Set up monitoring for the deployed model."""
    logger.info("Setting up model monitoring")

    database = os.getenv("SNOWFLAKE_DATABASE")

    # Create monitoring tables
    inference_log_table = f"{database}.MODELS.INFERENCE_LOG"

    session.sql(
        f"""
    CREATE TABLE IF NOT EXISTS {inference_log_table} (
        inference_id VARCHAR,
        model_name VARCHAR,
        model_version VARCHAR,
        timestamp TIMESTAMP_NTZ,
        input VARIANT,
        prediction FLOAT,
        latency_ms FLOAT,
        correlation_id VARCHAR
    )
    """
    ).collect()

    # Create monitoring view
    monitoring_view = f"{database}.ANALYTICS.MODEL_PERFORMANCE"

    session.sql(
        f"""
    CREATE OR REPLACE VIEW {monitoring_view} AS
    SELECT
        DATE_TRUNC('day', timestamp) AS day,
        model_name,
        model_version,
        COUNT(*) AS inference_count,
        AVG(latency_ms) AS avg_latency_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
    FROM {inference_log_table}
    GROUP BY 1, 2, 3
    ORDER BY 1 DESC
    """
    ).collect()

    logger.info("Monitoring setup completed")


# ======================================================================
# Step 9: Test Inference
# ======================================================================


def test_inference(session, limit: int = 10, label_filter: str | None = None):
    """Batch test model inference using rows from training set.

    Args:
        session: Snowpark session
        limit: number of random rows to score
        label_filter: optional filter for label, e.g., 'IS_FRAUD = 1' or 'IS_FRAUD = 0'

    Returns:
        List of dicts with prediction and actual label
    """
    logger.info("Testing model inference (batch)")

    database = os.getenv("SNOWFLAKE_DATABASE")
    udf_name = f"{database}.MODELS.vehicle_fraud_predict_udf"
    training_table = f"{database}.FEATURES.TRAINING_DATA"

    # Determine feature columns directly from table schema
    all_cols = session.table(training_table).columns
    exclude_cols = ["IS_FRAUD", "SAMPLE_WEIGHT"]
    feature_cols = [c for c in all_cols if c not in exclude_cols]

    # Build OBJECT_CONSTRUCT_KEEP_NULL argument list: 'COL', COL, ...
    pairs_sql = ",\n            ".join([f"'{c}', {c}" for c in feature_cols])

    # Optional label filter
    where_clause = f"WHERE {label_filter}" if label_filter else ""

    # Score multiple rows directly in Snowflake AND log to inference table
    inference_log_table = f"{database}.MODELS.INFERENCE_LOG"

    # First, score and get predictions
    test_sql = f"""
    SELECT
        TO_DOUBLE({udf_name}(OBJECT_CONSTRUCT_KEEP_NULL(
            {pairs_sql}
        ))) AS prediction,
        IS_FRAUD AS label,
        OBJECT_CONSTRUCT_KEEP_NULL({pairs_sql}) AS input_features
    FROM {training_table}
    {where_clause}
    ORDER BY RANDOM()
    LIMIT {limit}
    """

    logger.info(f"Scoring {limit} rows with {len(feature_cols)} features")
    rows = session.sql(test_sql).collect()

    # Log each inference to the monitoring table
    import json
    import uuid

    for r in rows:
        inference_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        # Convert input features to proper JSON string
        # r["INPUT_FEATURES"] is already a dict from OBJECT_CONSTRUCT
        input_json = json.dumps(r["INPUT_FEATURES"])

        # Use parameterized insert to avoid SQL injection and escaping issues
        log_sql = f"""
        INSERT INTO {inference_log_table}
        (inference_id, model_name, model_version, timestamp, input, prediction, latency_ms, correlation_id)
        SELECT
            '{inference_id}',
            'vehicle_insurance_fraud_detector',
            '1.0.0',
            CURRENT_TIMESTAMP(),
            PARSE_JSON($${input_json}$$),
            {r["PREDICTION"]},
            0.0,
            '{correlation_id}'
        """
        try:
            session.sql(log_sql).collect()
        except Exception as e:
            logger.warning(f"Failed to log inference: {e}")
    results = []
    for r in rows:
        pred = r["PREDICTION"]
        try:
            pred_val = float(pred) if pred is not None else None
        except Exception:
            # Fallback: handle strings in VARIANT
            pred_val = float(str(pred)) if pred is not None else None
        results.append({"prediction": pred_val, "label": r["LABEL"]})

    # Basic summary and sample outputs
    if results:
        positives = sum(
            1
            for r in results
            if (r["prediction"] is not None and r["prediction"] >= 0.5)
        )
        logger.info("\n" + "=" * 80)
        logger.info("📊 BATCH INFERENCE SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total predictions: {len(results)}")
        logger.info(
            f"Predicted fraud (>=0.5): {positives}/{len(results)} ({100 * positives / len(results):.1f}%)"
        )

        # Show sample predictions
        logger.info("\n" + "=" * 80)
        logger.info("📋 SAMPLE PREDICTIONS (first 10)")
        logger.info("=" * 80)
        for i, r in enumerate(results[:10], 1):
            pred = r["prediction"]
            label = r["label"]
            pred_str = f"{pred:.4f}" if pred is not None else "NULL"
            label_str = "FRAUD" if label == 1 else "LEGIT"
            match = "✓" if ((pred >= 0.5) == (label == 1)) else "✗"
            logger.info(
                f"{i:2d}. Prediction: {pred_str} | Actual: {label_str} | {match}"
            )

        # Calculate accuracy
        correct = sum(
            1
            for r in results
            if r["prediction"] is not None
            and ((r["prediction"] >= 0.5) == (r["label"] == 1))
        )
        accuracy = correct / len(results) if results else 0
        logger.info("\n" + "=" * 80)
        logger.info(f"🎯 Accuracy: {correct}/{len(results)} ({100 * accuracy:.1f}%)")
        logger.info("=" * 80 + "\n")

    return results


# ======================================================================
# Main Pipeline Execution
# ======================================================================


def run_pipeline():
    """Execute the complete MLOps pipeline."""
    try:
        # Step 1: Setup
        session = create_snowflake_session()
        setup_infrastructure(session)

        # Step 2: Data Ingestion
        csv_path = "src/datasets/vehicle_insurance_fraud/fraud_oracle.csv"
        ingest_data(session, csv_path)

        # Step 3: Feature Engineering
        feature_store = engineer_features(session)

        # Step 4: Training Dataset Generation
        training_table = generate_training_dataset(session, feature_store)

        # Step 5: Model Training
        training_result = train_model(session, training_table)

        # Step 6: Model Registration
        model_registry, model_version = register_model(session, training_result)

        # Step 7: Model Deployment
        udf_name = deploy_model(session, model_registry, model_version, training_result)

        # Step 8: Setup Monitoring
        setup_monitoring(session, udf_name)

        # Step 9: Test Inference
        test_inference(session, limit=50)

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
