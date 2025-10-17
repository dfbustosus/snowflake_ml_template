"""Entity definitions for the Feature Store.

This module defines Entity and SQLEntity classes that represent business
entities in the feature store.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from snowflake.snowpark import DataFrame


@dataclass
class Entity:
    """Represents a business entity in the feature store."""

    name: str
    join_keys: List[str]
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate entity attributes."""
        if not self.name:
            raise ValueError("Entity name cannot be empty")
        if not self.join_keys:
            raise ValueError("Entity must have at least one join key")
        if len(set(self.join_keys)) != len(self.join_keys):
            raise ValueError("Join keys must be unique")

    def get_join_key_columns(self) -> List[str]:
        """Get the join key columns."""
        return self.join_keys.copy()

    def validate_join_keys(self, df: DataFrame) -> None:
        """Validate join keys."""
        missing_keys = [key for key in self.join_keys if key not in df.columns]
        if missing_keys:
            raise ValueError(f"Missing join keys in DataFrame: {missing_keys}")


@dataclass
class SQLEntity:
    """Represents a business entity defined by SQL."""

    name: str
    join_keys: List[str]
    sql_definition: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
