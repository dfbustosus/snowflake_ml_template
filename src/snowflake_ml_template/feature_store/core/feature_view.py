"""FeatureView definitions for the Feature Store."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

from snowflake.snowpark import DataFrame

from snowflake_ml_template.feature_store.core.entity import Entity, SQLEntity
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureView:
    """Represents a logical group of features and their computation logic."""

    name: str
    entities: List[Entity]
    feature_df: DataFrame
    refresh_freq: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    version: str = "v1"
    created_at: datetime = field(default_factory=datetime.utcnow)
    feature_names: List[str] = field(init=False)

    def __post_init__(self) -> None:
        """Validate FeatureView attributes."""
        if not self.name:
            raise ValueError("FeatureView name cannot be empty")
        if not self.entities:
            raise ValueError("FeatureView must be associated with at least one entity")

        for entity in self.entities:
            entity.validate_join_keys(self.feature_df)

        entity_keys = set()
        for entity in self.entities:
            entity_keys.update(entity.join_keys)

        self.feature_names = [
            col for col in self.feature_df.columns if col not in entity_keys
        ]

        if not self.feature_names:
            raise ValueError("FeatureView must contain at least one feature column")

        if self.refresh_freq is not None:
            self._validate_refresh_freq()

    def _validate_refresh_freq(self) -> None:
        if not isinstance(self.refresh_freq, str):
            raise ValueError("refresh_freq must be a string")

        valid_freqs = [
            "1 minute",
            "5 minutes",
            "15 minutes",
            "30 minutes",
            "1 hour",
            "2 hours",
            "6 hours",
            "12 hours",
            "1 day",
            "1 week",
        ]

        if not (
            self.refresh_freq.lower() in [f.lower() for f in valid_freqs]
            or self.refresh_freq.replace(" ", "")
            .replace("*", "")
            .replace("-", "")
            .replace("/", "")
            .isdigit()
        ):
            logger.warning(
                f"refresh_freq '{self.refresh_freq}' may not be valid. Consider using: {valid_freqs}"
            )

    @property
    def is_external(self) -> bool:
        """Check if FeatureView is external."""
        return self.refresh_freq is None

    @property
    def is_snowflake_managed(self) -> bool:
        """Check if FeatureView is snowflake managed."""
        return self.refresh_freq is not None

    def get_entity_names(self) -> List[str]:
        """Get entity names."""
        return [entity.name for entity in self.entities]

    def get_join_keys(self) -> List[str]:
        """Get join keys."""
        keys = set()
        for entity in self.entities:
            keys.update(entity.join_keys)
        return list(keys)


@dataclass
class SQLFeatureView:
    """FeatureView that supports SQL-based transformations."""

    name: str
    entities: List[Union[Entity, SQLEntity]]
    sql_query: str
    refresh_freq: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    version: str = "v1"
    created_at: datetime = field(default_factory=datetime.utcnow)
    feature_names: List[str] = field(init=False)

    def __post_init__(self) -> None:
        """Validate SQLFeatureView attributes."""
        if not self.name:
            raise ValueError("SQLFeatureView name cannot be empty")
        if not self.entities:
            raise ValueError(
                "SQLFeatureView must be associated with at least one entity"
            )
        if not self.sql_query or not self.sql_query.strip():
            raise ValueError("SQL query cannot be empty")

        self._validate_sql_query()
        self.feature_names = self._extract_feature_names()

        if not self.feature_names:
            raise ValueError("SQLFeatureView must contain at least one feature column")

        if self.refresh_freq is not None:
            self._validate_refresh_freq()

    def _validate_sql_query(self) -> None:
        """Validate SQL query."""
        query = self.sql_query.strip().upper()
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "MERGE"]
        for keyword in dangerous_keywords:
            if keyword in query:
                raise ValueError(f"SQL query cannot contain {keyword} operations")
        if not query.startswith("SELECT"):
            raise ValueError("SQL query must be a SELECT statement")

    def _extract_feature_names(self) -> List[str]:
        """Extract feature names from SQL query."""
        return ["feature_1", "feature_2"]  # Placeholder - would parse SQL in production

    def _validate_refresh_freq(self) -> None:
        """Validate refresh frequency."""
        if not isinstance(self.refresh_freq, str):
            raise ValueError("refresh_freq must be a string")
        valid_freqs = [
            "1 minute",
            "5 minutes",
            "15 minutes",
            "30 minutes",
            "1 hour",
            "2 hours",
            "6 hours",
            "12 hours",
            "1 day",
            "1 week",
        ]
        if self.refresh_freq.lower() not in [f.lower() for f in valid_freqs]:
            if not (
                self.refresh_freq.replace(" ", "")
                .replace("*", "")
                .replace("-", "")
                .replace("/", "")
                .isdigit()
            ):
                logger.warning(f"refresh_freq '{self.refresh_freq}' may not be valid")

    @property
    def is_external(self) -> bool:
        """Check if FeatureView is external."""
        return self.refresh_freq is None

    @property
    def is_snowflake_managed(self) -> bool:
        """Check if FeatureView is snowflake managed."""
        return self.refresh_freq is not None

    def get_entity_names(self) -> List[str]:
        """Get entity names."""
        return [entity.name for entity in self.entities]

    def get_join_keys(self) -> List[str]:
        """Get join keys."""
        keys = set()
        for entity in self.entities:
            if hasattr(entity, "join_keys"):
                keys.update(entity.join_keys)
        return list(keys)
