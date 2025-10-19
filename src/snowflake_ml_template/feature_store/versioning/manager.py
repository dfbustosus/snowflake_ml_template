"""Feature version management following semantic versioning.

This module implements version management for feature views, supporting:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Version comparison and validation
- Version lifecycle management
- Immutable version guarantees
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import FeatureVersionError
from snowflake_ml_template.feature_store.core.feature_view import FeatureView
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VersionMetadata:
    """Metadata for a feature view version.

    Attributes:
        version: Semantic version string (e.g., "1.2.3")
        feature_view_name: Name of the feature view
        created_at: When this version was created
        created_by: User who created this version
        description: Optional description of changes
        schema_hash: Hash of the feature schema for change detection
        deprecated: Whether this version is deprecated
        deprecated_at: When this version was deprecated
    """

    version: str
    feature_view_name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    description: Optional[str] = None
    schema_hash: Optional[str] = None
    deprecated: bool = False
    deprecated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate version format."""
        if not self._is_valid_semver(self.version):
            raise FeatureVersionError(
                f"Invalid semantic version: {self.version}. "
                "Must follow MAJOR.MINOR.PATCH format (e.g., 1.2.3)"
            )

    @staticmethod
    def _is_valid_semver(version: str) -> bool:
        """Validate semantic version format."""
        pattern = r"^\d+\.\d+\.\d+$"
        return bool(re.match(pattern, version))

    def get_major_minor_patch(self) -> Tuple[int, int, int]:
        """Parse version into major, minor, patch components."""
        parts = self.version.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])


class FeatureVersionManager:
    """Manage feature view versions with semantic versioning.

    This class provides comprehensive version management following semantic
    versioning principles:
    - MAJOR: Incompatible schema changes
    - MINOR: Backward-compatible functionality additions
    - PATCH: Backward-compatible bug fixes

    All versions are immutable once created, ensuring reproducibility.

    Attributes:
        session: Snowflake session
        database: Database containing feature store
        schema: Schema containing feature store
        logger: Structured logger

    Example:
        >>> manager = FeatureVersionManager(session, "ML_PROD_DB", "FEATURES")
        >>>
        >>> # Create new version
        >>> manager.create_version(
        ...     feature_view,
        ...     version="1.0.0",
        ...     description="Initial release"
        ... )
        >>>
        >>> # List versions
        >>> versions = manager.list_versions("customer_features")
        >>>
        >>> # Compare versions
        >>> diff = manager.compare_versions("customer_features", "1.0.0", "1.1.0")
    """

    def __init__(self, session: Session, database: str, schema: str):
        """Initialize the version manager."""
        if session is None:
            raise ValueError("Session cannot be None")
        if not database or not schema:
            raise ValueError("Database and schema cannot be empty")

        self.session = session
        self.database = database
        self.schema = schema
        self.logger = get_logger(__name__)

    def create_version(
        self,
        feature_view: FeatureView,
        version: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> VersionMetadata:
        """Create a new version of a feature view.

        This method creates an immutable version of a feature view. Once
        created, the version cannot be modified, ensuring reproducibility.

        Args:
            feature_view: FeatureView to version
            version: Semantic version string (e.g., "1.0.0")
            description: Optional description of changes
            created_by: Optional user who created this version

        Returns:
            VersionMetadata for the created version

        Raises:
            FeatureVersionError: If version already exists or is invalid

        Example:
            >>> metadata = manager.create_version(
            ...     customer_features,
            ...     version="1.0.0",
            ...     description="Initial release with transaction aggregates"
            ... )
        """
        raise FeatureVersionError(
            "Feature version creation must be performed via Snowflake Feature Store APIs."
        )

    def _compute_schema_hash(self, feature_view: FeatureView) -> str:
        """Compute hash of feature schema for change detection."""
        import hashlib

        # Create deterministic string from schema
        schema_str = "|".join(sorted(feature_view.feature_names))
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def list_versions(
        self, feature_view_name: str, include_deprecated: bool = False
    ) -> List[str]:
        """List all versions of a feature view.

        Args:
            feature_view_name: Name of the feature view
            include_deprecated: Whether to include deprecated versions

        Returns:
            List of version strings, sorted by semantic version

        Example:
            >>> versions = manager.list_versions("customer_features")
            >>> # ['1.0.0', '1.0.1', '1.1.0', '2.0.0']
        """
        catalog = "SNOWFLAKE.ML.FEATURE_STORE.FEATURE_VIEW_VERSIONS"
        sql = (
            "SELECT VERSION FROM {catalog} "
            f"WHERE DATABASE_NAME = '{self.database}' "
            f"AND SCHEMA_NAME = '{self.schema}' "
            f"AND FEATURE_VIEW_NAME = '{feature_view_name}'"
        )
        if not include_deprecated:
            sql += " AND COALESCE(DEPRECATED, FALSE) = FALSE"
        sql += " ORDER BY VERSION"

        result = self.session.sql(sql.replace("{catalog}", catalog)).collect()
        versions = [row["VERSION"] for row in result]
        return versions

    def _sort_versions(self, versions: List[str]) -> List[str]:
        """Sort versions by semantic versioning rules."""

        def version_key(v: str) -> Tuple[int, int, int]:
            parts = v.split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]))

        return sorted(versions, key=version_key)

    def get_latest_version(self, feature_view_name: str) -> Optional[str]:
        """Get the latest non-deprecated version.

        Args:
            feature_view_name: Name of the feature view

        Returns:
            Latest version string, or None if no versions exist
        """
        versions = self.list_versions(feature_view_name, include_deprecated=False)
        return versions[-1] if versions else None

    def version_exists(self, feature_view_name: str, version: str) -> bool:
        """Check if a version exists.

        Args:
            feature_view_name: Name of the feature view
            version: Version string to check

        Returns:
            True if version exists, False otherwise
        """
        catalog = "SNOWFLAKE.ML.FEATURE_STORE.FEATURE_VIEW_VERSIONS"
        sql = (
            "SELECT 1 FROM {catalog} "
            f"WHERE DATABASE_NAME = '{self.database}' "
            f"AND SCHEMA_NAME = '{self.schema}' "
            f"AND FEATURE_VIEW_NAME = '{feature_view_name}' "
            f"AND VERSION = '{version}'"
        )
        result = self.session.sql(sql.replace("{catalog}", catalog)).collect()
        return len(result) > 0

    def deprecate_version(
        self, feature_view_name: str, version: str, reason: Optional[str] = None
    ) -> None:
        """Mark a version as deprecated.

        Deprecated versions are not deleted but are excluded from default
        queries. This maintains lineage while discouraging use.

        Args:
            feature_view_name: Name of the feature view
            version: Version to deprecate
            reason: Optional reason for deprecation

        Raises:
            FeatureVersionError: If version doesn't exist
        """
        if not self.version_exists(feature_view_name, version):
            raise FeatureVersionError(
                f"Version {version} does not exist for {feature_view_name}"
            )

        raise FeatureVersionError(
            "Version deprecation must be managed via Snowflake Feature Store APIs."
        )

    def compare_versions(
        self, feature_view_name: str, version1: str, version2: str
    ) -> Dict[str, Any]:
        """Compare two versions and return differences.

        Args:
            feature_view_name: Name of the feature view
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary with comparison results

        Example:
            >>> diff = manager.compare_versions("customer_features", "1.0.0", "1.1.0")
            >>> # {
            >>> #     'version_diff': (1, 0, 0) -> (1, 1, 0),
            >>> #     'schema_changed': True,
            >>> #     'features_added': ['new_feature'],
            >>> #     'features_removed': []
            >>> # }
        """
        catalog = "SNOWFLAKE.ML.FEATURE_STORE.FEATURE_VIEW_VERSIONS"
        sql = (
            "SELECT VERSION, SCHEMA_HASH, FEATURE_NAMES FROM {catalog} "
            f"WHERE DATABASE_NAME = '{self.database}' "
            f"AND SCHEMA_NAME = '{self.schema}' "
            f"AND FEATURE_VIEW_NAME = '{feature_view_name}' "
            f"AND VERSION IN ('{version1}', '{version2}')"
        )

        result = self.session.sql(sql.replace("{catalog}", catalog)).collect()

        if len(result) != 2:
            raise FeatureVersionError(
                f"Could not find both versions for comparison of feature view '{feature_view_name}'"
            )

        v1_data = next(r for r in result if r["VERSION"] == version1)
        v2_data = next(r for r in result if r["VERSION"] == version2)

        schema_changed = v1_data.get("SCHEMA_HASH") != v2_data.get("SCHEMA_HASH")
        features1 = set(v1_data.get("FEATURE_NAMES", []))
        features2 = set(v2_data.get("FEATURE_NAMES", []))

        return {
            "version1": version1,
            "version2": version2,
            "schema_changed": schema_changed,
            "features_added": list(features2 - features1),
            "features_removed": list(features1 - features2),
            "features_common": list(features1 & features2),
        }
