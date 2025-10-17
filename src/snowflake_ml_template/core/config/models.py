"""Pydantic models for type-safe configuration.

This module defines Pydantic models for all configuration types in the
MLOps framework. These models provide:
- Type safety with automatic validation
- Environment variable substitution
- Default values
- Documentation via field descriptions

Models:
    SnowflakeConfig: Snowflake connection configuration
    FeatureStoreConfig: Feature store configuration
    ModelRegistryConfig: Model registry configuration
    MonitoringConfig: Monitoring configuration
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SnowflakeConfig(BaseModel):
    """Snowflake connection configuration.

    This model defines all parameters needed to connect to Snowflake.
    It supports both password and key-pair authentication.

    Attributes:
        account: Snowflake account identifier
        user: Snowflake username
        password: Optional password for authentication
        authenticator: Optional authenticator (e.g., 'externalbrowser')
        private_key_path: Optional path to private key file
        warehouse: Default warehouse to use
        database: Default database to use
        schema: Default schema to use
        role: Optional role to use
        session_parameters: Optional session parameters
    """

    account: str = Field(..., description="Snowflake account identifier")
    user: str = Field(..., description="Snowflake username")
    password: Optional[str] = Field(None, description="Password for authentication")
    authenticator: Optional[str] = Field(None, description="Authentication method")
    private_key_path: Optional[str] = Field(
        None, description="Path to private key file"
    )
    warehouse: str = Field(..., description="Default warehouse")
    database: str = Field(..., description="Default database")
    schema_: str = Field(
        "PUBLIC",
        alias="schema",
        validation_alias="schema",
        serialization_alias="schema",
        description="Default schema",
    )
    role: Optional[str] = Field(None, description="Role to use")
    session_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional session parameters"
    )

    @field_validator("account")
    @classmethod
    def validate_account(cls, v: str) -> str:
        """Validate account identifier.

        Args:
            v: Account identifier to validate

        Returns:
            Validated account identifier

        Raises:
            ValueError: If account identifier is invalid
        """
        if not v or "." not in v:
            raise ValueError(
                "Account must be in format <account_identifier>.<region>.<cloud>"
            )
        return v

    @field_validator("user")
    @classmethod
    def validate_user(cls, v: str) -> str:
        """Validate username.

        Args:
            v: Username to validate

        Returns:
            Validated username

        Raises:
            ValueError: If username is empty
        """
        if not v:
            raise ValueError("User cannot be empty")
        return v.strip()

    # Pydantic v2 configuration
    model_config = ConfigDict(
        extra="allow",
        protected_namespaces=(),
        populate_by_name=True,
    )


class FeatureStoreConfig(BaseModel):
    """Feature store configuration.

    Attributes:
        database: Database containing feature store
        schema: Schema containing feature store
        metadata_table: Table for feature metadata
        enable_versioning: Whether to enable feature versioning
        enable_lineage: Whether to track feature lineage
        enable_monitoring: Whether to enable feature monitoring
        refresh_frequency: Default refresh frequency for feature views
    """

    database: str = Field(..., description="Feature store database")
    schema_: str = Field(
        "FEATURES",
        alias="schema",
        validation_alias="schema",
        serialization_alias="schema",
        description="Feature store schema",
    )
    metadata_table: str = Field(
        "FEATURE_METADATA", description="Table for feature metadata"
    )
    enable_versioning: bool = Field(True, description="Enable feature versioning")
    enable_lineage: bool = Field(True, description="Track feature lineage")
    enable_monitoring: bool = Field(True, description="Enable feature monitoring")
    refresh_frequency: str = Field("1 day", description="Default refresh frequency")

    model_config = ConfigDict(
        extra="forbid", protected_namespaces=(), populate_by_name=True
    )


class ModelRegistryConfig(BaseModel):
    """Model registry configuration.

    Attributes:
        database: Database containing model registry
        schema: Schema containing model registry
        models_table: Table for model metadata
        versions_table: Table for model versions
        stage_path: Snowflake stage for model artifacts
        enable_lineage: Whether to track model lineage
        enable_promotion_workflow: Whether to enable promotion workflow
        default_version_alias: Default version alias name
    """

    database: str = Field(..., description="Model registry database")
    schema_: str = Field(
        "MODELS",
        alias="schema",
        validation_alias="schema",
        serialization_alias="schema",
        description="Model registry schema",
    )
    models_table: str = Field("MODEL_METADATA", description="Models metadata table")
    versions_table: str = Field("MODEL_VERSIONS", description="Model versions table")
    stage_path: str = Field("@ML_MODELS_STAGE", description="Stage for model artifacts")
    enable_lineage: bool = Field(True, description="Track model lineage")
    enable_promotion_workflow: bool = Field(
        True, description="Enable promotion workflow"
    )
    default_version_alias: str = Field("DEFAULT", description="Default version alias")

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class MonitoringConfig(BaseModel):
    """Monitoring configuration.

    Attributes:
        database: Database for monitoring data
        schema: Schema for monitoring data
        enable_model_monitoring: Whether to enable model monitoring
        enable_data_monitoring: Whether to enable data monitoring
        enable_cost_monitoring: Whether to enable cost monitoring
        drift_threshold: Threshold for drift detection (PSI)
        performance_threshold: Threshold for performance degradation
        alert_channels: List of alert channels (email, slack, etc.)
    """

    database: str = Field(..., description="Monitoring database")
    schema_: str = Field(
        "MONITORING",
        alias="schema",
        validation_alias="schema",
        serialization_alias="schema",
        description="Monitoring schema",
    )
    enable_model_monitoring: bool = Field(True, description="Enable model monitoring")
    enable_data_monitoring: bool = Field(True, description="Enable data monitoring")
    enable_cost_monitoring: bool = Field(True, description="Enable cost monitoring")
    drift_threshold: float = Field(0.1, description="Drift detection threshold (PSI)")
    performance_threshold: float = Field(
        0.05, description="Performance degradation threshold"
    )
    alert_channels: List[str] = Field(
        default_factory=list, description="Alert channels"
    )

    @field_validator("drift_threshold")
    @classmethod
    def validate_drift_threshold(cls, v: float) -> float:
        """Validate drift threshold."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Drift threshold must be between 0.0 and 1.0")
        return v

    @field_validator("performance_threshold")
    @classmethod
    def validate_performance_threshold(cls, v: float) -> float:
        """Validate performance threshold."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Performance threshold must be between 0.0 and 1.0")
        return v

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class IngestionConfigModel(BaseModel):
    """Data ingestion configuration.

    Attributes:
        method: Ingestion method (snowpipe, copy_into, streaming)
        source_type: Source type (s3, azure_blob, gcs, etc.)
        source_location: Source location URI
        file_format: File format (CSV, JSON, PARQUET, etc.)
        target_database: Target database
        target_schema: Target schema
        target_table: Target table
        warehouse: Warehouse for ingestion
        on_error: Error handling strategy
        purge: Whether to purge files after ingestion
    """

    method: str = Field(..., description="Ingestion method")
    source_type: str = Field(..., description="Source type")
    source_location: str = Field(..., description="Source location")
    file_format: str = Field(..., description="File format")
    target_database: str = Field(..., description="Target database")
    target_schema: str = Field(..., description="Target schema")
    target_table: str = Field(..., description="Target table")
    warehouse: str = Field(..., description="Warehouse for ingestion")
    on_error: str = Field("ABORT_STATEMENT", description="Error handling strategy")
    purge: bool = Field(False, description="Purge files after ingestion")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate ingestion method.

        Args:
            v: The ingestion method to validate

        Returns:
            The validated ingestion method in lowercase

        Raises:
            ValueError: If the method is not one of the valid methods
        """
        valid_methods = {"snowpipe", "copy_into", "streaming"}
        if v.lower() not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v.lower()

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class TransformationConfigModel(BaseModel):
    """Data transformation configuration.

    Attributes:
        transformation_type: Type of transformation (snowpark, sql, dbt)
        source_database: Source database
        source_schema: Source schema
        source_table: Source table
        target_database: Target database
        target_schema: Target schema
        target_table: Target table
        warehouse: Warehouse for transformation
        mode: Write mode (overwrite, append, merge)
    """

    transformation_type: str = Field(..., description="Transformation type")
    source_database: str = Field(..., description="Source database")
    source_schema: str = Field(..., description="Source schema")
    source_table: str = Field(..., description="Source table")
    target_database: str = Field(..., description="Target database")
    target_schema: str = Field(..., description="Target schema")
    target_table: str = Field(..., description="Target table")
    warehouse: str = Field(..., description="Warehouse for transformation")
    mode: str = Field("overwrite", description="Write mode")

    @field_validator("transformation_type")
    @classmethod
    def validate_transformation_type(cls, v: str) -> str:
        """Validate transformation type.

        Args:
            v: The transformation type to validate

        Returns:
            The validated transformation type in lowercase

        Raises:
            ValueError: If the transformation type is not valid
        """
        valid_types = ["snowpark", "sql", "dbt", "dynamic_table"]
        if v.lower() not in valid_types:
            raise ValueError(f"Transformation type must be one of: {valid_types}")
        return v.lower()

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate write mode.

        Args:
            v: The write mode to validate

        Returns:
            The validated write mode in lowercase

        Raises:
            ValueError: If the write mode is not valid
        """
        valid_modes = ["overwrite", "append", "merge"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Mode must be one of: {valid_modes}")
        return v.lower()

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class TrainingConfigModel(BaseModel):
    """Model training configuration.

    Attributes:
        strategy: Training strategy (single_node, distributed, gpu, many_model)
        framework: ML framework (sklearn, xgboost, lightgbm, pytorch, tensorflow)
        model_type: Type of model (classifier, regressor, etc.)
        training_database: Database containing training data
        training_schema: Schema containing training data
        training_table: Training data table
        warehouse: Warehouse for training
        hyperparameters: Model hyperparameters
        validation_split: Validation split fraction
        test_split: Test split fraction
        random_state: Random seed for reproducibility
    """

    strategy: str = Field(..., description="Training strategy")
    framework: str = Field(..., description="ML framework")
    model_type: str = Field(..., description="Model type")
    training_database: str = Field(..., description="Training database")
    training_schema: str = Field(..., description="Training schema")
    training_table: str = Field(..., description="Training table")
    warehouse: str = Field(..., description="Warehouse for training")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    validation_split: float = Field(0.2, description="Validation split fraction")
    test_split: float = Field(0.1, description="Test split fraction")
    random_state: int = Field(42, description="Random seed")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate training strategy.

        Args:
            v: The training strategy to validate

        Returns:
            The validated training strategy in lowercase

        Raises:
            ValueError: If the training strategy is not valid
        """
        valid_strategies = ["single_node", "distributed", "gpu", "many_model"]
        if v.lower() not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        return v.lower()

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        """Validate ML framework.

        Args:
            v: The ML framework to validate

        Returns:
            The validated ML framework in lowercase

        Raises:
            ValueError: If the ML framework is not valid
        """
        valid_frameworks = ["sklearn", "xgboost", "lightgbm", "pytorch", "tensorflow"]
        if v.lower() not in valid_frameworks:
            raise ValueError(f"Framework must be one of: {valid_frameworks}")
        return v.lower()

    @field_validator("validation_split", "test_split")
    @classmethod
    def validate_split(cls, v: float) -> float:
        """Validate split fraction.

        Args:
            v: The split fraction to validate

        Returns:
            The validated split fraction

        Raises:
            ValueError: If the split fraction is not between 0 and 1
        """
        if not (0.0 < v < 1.0):
            raise ValueError("Split must be between 0 and 1")
        return v

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class DeploymentConfigModel(BaseModel):
    """Model deployment configuration.

    Attributes:
        strategy: Deployment strategy (warehouse_udf, spcs, external)
        target: Deployment target (batch, realtime, streaming)
        model_name: Name of model to deploy
        model_version: Version of model to deploy
        deployment_database: Target database for deployment
        deployment_schema: Target schema for deployment
        deployment_name: Name for deployed model/service
        warehouse: Warehouse for deployment
        compute_pool: Compute pool for SPCS (if applicable)
        instance_count: Number of instances
        enable_monitoring: Whether to enable monitoring
    """

    strategy: str = Field(..., description="Deployment strategy")
    target: str = Field(..., description="Deployment target")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    deployment_database: str = Field(..., description="Deployment database")
    deployment_schema: str = Field(..., description="Deployment schema")
    deployment_name: str = Field(..., description="Deployment name")
    warehouse: str = Field(..., description="Warehouse for deployment")
    compute_pool: Optional[str] = Field(None, description="Compute pool for SPCS")
    instance_count: int = Field(1, description="Number of instances")
    enable_monitoring: bool = Field(True, description="Enable monitoring")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate deployment strategy.

        Args:
            v: The deployment strategy to validate

        Returns:
            The validated deployment strategy in lowercase

        Raises:
            ValueError: If the deployment strategy is not valid
        """
        valid_strategies = ["warehouse_udf", "spcs", "external"]
        if v.lower() not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        return v.lower()

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        """Validate deployment target.

        Args:
            v: The deployment target to validate

        Returns:
            The validated deployment target in lowercase

        Raises:
            ValueError: If the deployment target is not valid
        """
        valid_targets = ["batch", "realtime", "streaming"]
        if v.lower() not in valid_targets:
            raise ValueError(f"Target must be one of: {valid_targets}")
        return v.lower()

    @field_validator("instance_count")
    @classmethod
    def validate_instance_count(cls, v: int) -> int:
        """Validate instance count.

        Args:
            v: The instance count to validate

        Returns:
            The validated instance count

        Raises:
            ValueError: If the instance count is less than 1
        """
        if v < 1:
            raise ValueError("Instance count must be >= 1")
        return v

    model_config = ConfigDict(extra="allow", protected_namespaces=())
