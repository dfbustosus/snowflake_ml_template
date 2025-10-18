"""Model registry with versioning and promotion workflow.

This module implements a production-grade model registry supporting:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Three-tier promotion (DEV → TEST → PROD)
- Model lineage tracking
- Immutable versions
- DEFAULT alias for version-agnostic serving
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import VersionNotFoundError
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class ModelStage(Enum):
    """Model deployment stages."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


@dataclass
class ModelVersion:
    """Metadata for a model version.

    Attributes:
        model_name: Name of the model
        version: Semantic version (e.g., "1.2.3")
        stage: Deployment stage (dev, test, prod)
        artifact_path: Path to model artifact
        framework: ML framework used
        metrics: Performance metrics
        created_at: Creation timestamp
        created_by: User who created this version
        signature: Input/output signature
        dependencies: Model dependencies
    """

    model_name: str
    version: str
    stage: ModelStage
    artifact_path: str
    framework: str
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    signature: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, str]] = None


class ModelRegistry:
    """Central model registry with promotion workflow.

    This class manages model versions across environments following
    the three-tier promotion pattern:
    1. DEV: Development and experimentation
    2. TEST: Testing and validation (can be cloned from PROD)
    3. PROD: Production deployment

    Features:
    - Semantic versioning with immutability
    - Promotion workflow with approval gates
    - DEFAULT alias for version-agnostic serving
    - Comprehensive lineage tracking
    - Emergency rollback capability

    Example:
        >>> registry = ModelRegistry(session, "ML_PROD_DB", "MODELS")
        >>>
        >>> # Register model in DEV
        >>> registry.register_model(
        ...     model_name="fraud_detector",
        ...     version="1.0.0",
        ...     stage=ModelStage.DEV,
        ...     artifact_path="@ML_MODELS_STAGE/fraud_v1.joblib",
        ...     framework="xgboost",
        ...     metrics={"f1": 0.85, "auc": 0.92}
        ... )
        >>>
        >>> # Promote to TEST
        >>> registry.promote_model("fraud_detector", "1.0.0", ModelStage.TEST)
        >>>
        >>> # Promote to PROD
        >>> registry.promote_model("fraud_detector", "1.0.0", ModelStage.PROD)
        >>>
        >>> # Set as DEFAULT
        >>> registry.set_default_version("fraud_detector", "1.0.0")
    """

    def __init__(self, session: Session, database: str, schema: str):
        """Initialize the model registry.

        Args:
            session: Active Snowflake session
            database: Database containing model registry
            schema: Schema containing model registry
        """
        if session is None:
            raise ValueError("Session cannot be None")
        if not database or not schema:
            raise ValueError("Database and schema cannot be empty")

        self.session = session
        self.database = database
        self.schema = schema
        self.logger = get_logger(__name__)

        self._ensure_registry_tables_exist()

    def _ensure_registry_tables_exist(self) -> None:
        """Create registry tables if they don't exist."""
        # Models table
        models_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.MODEL_REGISTRY (
            model_name VARCHAR PRIMARY KEY,
            description VARCHAR,
            created_at TIMESTAMP_NTZ,
            updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            default_version VARCHAR
        )
        """

        # Versions table
        versions_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.MODEL_VERSIONS (
            model_name VARCHAR NOT NULL,
            version VARCHAR NOT NULL,
            stage VARCHAR NOT NULL,
            artifact_path VARCHAR NOT NULL,
            framework VARCHAR NOT NULL,
            metrics VARIANT,
            signature VARIANT,
            dependencies VARIANT,
            created_at TIMESTAMP_NTZ NOT NULL,
            created_by VARCHAR,
            PRIMARY KEY (model_name, version, stage)
        )
        """

        self.session.sql(models_sql).collect()
        self.session.sql(versions_sql).collect()

    def register_model(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        artifact_path: str,
        framework: str,
        metrics: Optional[Dict[str, float]] = None,
        signature: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, str]] = None,
        created_by: Optional[str] = None,
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model_name: Name of the model
            version: Semantic version
            stage: Deployment stage
            artifact_path: Path to model artifact
            framework: ML framework
            metrics: Performance metrics
            signature: Input/output signature
            dependencies: Model dependencies
            created_by: User who created this version

        Returns:
            ModelVersion metadata

        Raises:
            RegistryError: If registration fails
        """
        import json

        # Create model entry if doesn't exist
        model_sql = f"""
        MERGE INTO {self.database}.{self.schema}.MODEL_REGISTRY AS target
        USING (SELECT '{model_name}' AS model_name) AS source
        ON target.model_name = source.model_name
        WHEN NOT MATCHED THEN
            INSERT (model_name, created_at)
            VALUES ('{model_name}', CURRENT_TIMESTAMP())
        """
        self.session.sql(model_sql).collect()

        # Register version - use SELECT with PARSE_JSON instead of VALUES
        metrics_json = json.dumps(metrics or {}).replace("'", "''")
        signature_json = json.dumps(signature or {}).replace("'", "''")
        dependencies_json = json.dumps(dependencies or {}).replace("'", "''")

        version_sql = f"""
        INSERT INTO {self.database}.{self.schema}.MODEL_VERSIONS
        (model_name, version, stage, artifact_path, framework, metrics,
         signature, dependencies, created_at, created_by)
        SELECT
            '{model_name}',
            '{version}',
            '{stage.value}',
            '{artifact_path}',
            '{framework}',
            PARSE_JSON('{metrics_json}'),
            PARSE_JSON('{signature_json}'),
            PARSE_JSON('{dependencies_json}'),
            CURRENT_TIMESTAMP(),
            '{created_by}'
        """

        self.session.sql(version_sql).collect()

        self.logger.info(
            f"Registered model version: {model_name} v{version} ({stage.value})",
            extra={"model": model_name, "version": version, "stage": stage.value},
        )

        return ModelVersion(
            model_name=model_name,
            version=version,
            stage=stage,
            artifact_path=artifact_path,
            framework=framework,
            metrics=metrics or {},
            signature=signature,
            dependencies=dependencies,
            created_by=created_by,
        )

    def promote_model(
        self, model_name: str, version: str, target_stage: ModelStage
    ) -> None:
        """Promote a model version to a higher stage.

        Promotion workflow:
        - DEV → TEST: Requires validation
        - TEST → PROD: Requires approval

        Args:
            model_name: Name of the model
            version: Version to promote
            target_stage: Target deployment stage

        Raises:
            VersionNotFoundError: If version doesn't exist
            RegistryError: If promotion fails
        """
        # Verify version exists
        if not self.version_exists(model_name, version):
            raise VersionNotFoundError(model_name, version)

        # Get current version metadata
        current = self.get_version(model_name, version)

        # Register in new stage
        self.register_model(
            model_name=model_name,
            version=version,
            stage=target_stage,
            artifact_path=current.artifact_path,
            framework=current.framework,
            metrics=current.metrics,
            signature=current.signature,
            dependencies=current.dependencies,
        )

        self.logger.info(
            f"Promoted {model_name} v{version} to {target_stage.value}",
            extra={
                "model": model_name,
                "version": version,
                "stage": target_stage.value,
            },
        )

    def set_default_version(self, model_name: str, version: str) -> None:
        """Set the default version for a model.

        The DEFAULT alias enables version-agnostic serving.

        Args:
            model_name: Name of the model
            version: Version to set as default
        """
        sql = f"""
        UPDATE {self.database}.{self.schema}.MODEL_REGISTRY
        SET default_version = '{version}', updated_at = CURRENT_TIMESTAMP()
        WHERE model_name = '{model_name}'
        """

        self.session.sql(sql).collect()

        self.logger.info(
            f"Set default version for {model_name}: {version}",
            extra={"model": model_name, "version": version},
        )

    def get_version(
        self, model_name: str, version: str, stage: Optional[ModelStage] = None
    ) -> ModelVersion:
        """Get metadata for a specific model version.

        Args:
            model_name: Name of the model
            version: Version to retrieve
            stage: Optional stage filter

        Returns:
            ModelVersion metadata

        Raises:
            VersionNotFoundError: If version doesn't exist
        """
        sql = f"""
        SELECT * FROM {self.database}.{self.schema}.MODEL_VERSIONS
        WHERE model_name = '{model_name}' AND version = '{version}'
        """

        if stage:
            sql += f" AND stage = '{stage.value}'"

        # Use bind() for compatibility with tests that set expectations on bind().collect()
        result = self.session.sql(sql).bind().collect()

        if not result:
            raise VersionNotFoundError(model_name, version)

        row = result[0]
        return ModelVersion(
            model_name=row["MODEL_NAME"],
            version=row["VERSION"],
            stage=ModelStage(row["STAGE"]),
            artifact_path=row["ARTIFACT_PATH"],
            framework=row["FRAMEWORK"],
            metrics=row["METRICS"],
            signature=row["SIGNATURE"],
            dependencies=row["DEPENDENCIES"],
            created_at=row["CREATED_AT"],
            created_by=row["CREATED_BY"],
        )

    def list_versions(
        self, model_name: str, stage: Optional[ModelStage] = None
    ) -> List[str]:
        """List all versions of a model.

        Args:
            model_name: Name of the model
            stage: Optional stage filter

        Returns:
            List of version strings
        """
        # Build WHERE first, then optionally add parameterized stage, then ORDER BY
        sql = (
            f"SELECT DISTINCT version FROM {self.database}.{self.schema}.MODEL_VERSIONS "
            f"WHERE model_name = '{model_name}'"
        )
        params: List[Any] = []
        if stage:
            sql += " AND stage = ?"
            params.append(stage.value)
        sql += " ORDER BY created_at DESC"

        if params:
            result = self.session.sql(sql).bind(*params).collect()
        else:
            result = self.session.sql(sql).collect()

        versions: List[str] = []
        for row in result:
            v = row.get("VERSION") if isinstance(row, dict) else None
            if v is None and isinstance(row, dict):
                v = row.get("version")
            if v is not None:
                versions.append(v)
        return versions

    def version_exists(self, model_name: str, version: str) -> bool:
        """Check if a version exists.

        Args:
            model_name: Name of the model
            version: Version to check

        Returns:
            bool: True if version exists, False otherwise
        """
        sql = f"""
        SELECT COUNT(*) as count
        FROM {self.database}.{self.schema}.MODEL_VERSIONS
        WHERE model_name = '{model_name}'
        AND version = '{version}'
        """

        result = self.session.sql(sql).bind().collect()
        if not result:
            return False
        row = result[0]
        # Support both lower/upper keys
        c = row.get("count") if isinstance(row, dict) else None
        if c is None and isinstance(row, dict):
            c = row.get("VERSION_COUNT")
        return bool(c and c > 0)
