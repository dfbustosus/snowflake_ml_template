"""Warehouse provisioning for Snowflake MLOps.

This module provides functionality to create and manage Snowflake warehouses
optimized for different ML workload types.

Classes:
    WarehouseProvisioner: Create and manage warehouses
"""

from typing import Dict, List, Optional, TypedDict

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import ConfigurationError
from snowflake_ml_template.utils.logging import get_logger


class WarehouseConfig(TypedDict):
    """Type definition for warehouse configuration."""

    size: str
    type: str
    auto_suspend: int
    min_cluster_count: int
    max_cluster_count: int
    comment: str


class WarehouseProvisioner:
    """Provision and manage Snowflake warehouses.

    This class handles warehouse creation following the Golden Migration Plan:
    - INGEST_WH: X-Small for data ingestion
    - TRANSFORM_WH: Medium-Large for transformations (multi-cluster)
    - ML_TRAINING_WH: Large Snowpark-Optimized for training
    - INFERENCE_WH: Medium for inference (multi-cluster)

    Attributes:
        session: Snowflake session
        logger: Structured logger

    Example:
        >>> provisioner = WarehouseProvisioner(session)
        >>> provisioner.create_mlops_warehouses()
    """

    MLOPS_WAREHOUSES = {
        "INGEST_WH": {
            "size": "X-SMALL",
            "type": "STANDARD",
            "auto_suspend": 60,
            "min_cluster_count": 1,
            "max_cluster_count": 1,
            "comment": "Warehouse for data ingestion",
        },
        "TRANSFORM_WH": {
            "size": "MEDIUM",
            "type": "STANDARD",
            "auto_suspend": 60,
            "min_cluster_count": 1,
            "max_cluster_count": 3,
            "comment": "Warehouse for data transformations",
        },
        "ML_TRAINING_WH": {
            "size": "LARGE",
            "type": "SNOWPARK-OPTIMIZED",
            "auto_suspend": 60,
            "min_cluster_count": 1,
            "max_cluster_count": 1,
            "comment": "Snowpark-Optimized warehouse for ML training",
        },
        "INFERENCE_WH": {
            "size": "MEDIUM",
            "type": "STANDARD",
            "auto_suspend": 60,
            "min_cluster_count": 1,
            "max_cluster_count": 2,
            "comment": "Warehouse for model inference",
        },
    }

    def __init__(self, session: Session) -> None:
        """Initialize the WarehouseProvisioner with a Snowflake session.

        Args:
            session: An active Snowflake session for executing warehouse operations.

        Raises:
            ValueError: If the provided session is None.
        """
        if session is None:
            raise ValueError("Session cannot be None")
        self.session = session
        self.logger = get_logger(__name__)

    def create_mlops_warehouses(self) -> Dict[str, bool]:
        """Create all standard MLOps warehouses.

        Returns:
            Dictionary mapping warehouse names to creation status (True if successful)
        """
        self.logger.info("Creating MLOps warehouses")
        results: Dict[str, bool] = {}

        for name, config in self.MLOPS_WAREHOUSES.items():
            # Create warehouse directly from config with type-safe access
            wh_config: WarehouseConfig = {
                "size": str(config["size"]),
                "type": str(config["type"]),
                "auto_suspend": int(str(config["auto_suspend"])),
                "min_cluster_count": int(str(config["min_cluster_count"])),
                "max_cluster_count": int(str(config["max_cluster_count"])),
                "comment": str(config["comment"]),
            }
            results[name] = self.create_warehouse(name=name, **wh_config)

        self.logger.info("MLOps warehouses created", extra={"results": results})
        return results

    def create_warehouse(
        self,
        name: str,
        size: str = "MEDIUM",
        type: str = "STANDARD",
        auto_suspend: int = 60,
        auto_resume: bool = True,
        min_cluster_count: int = 1,
        max_cluster_count: int = 1,
        comment: Optional[str] = None,
    ) -> bool:
        """Create a warehouse if it doesn't exist."""
        if not name:
            raise ValueError("Warehouse name cannot be empty")

        self.logger.info(f"Creating warehouse: {name}")

        try:
            sql_parts = [f"CREATE WAREHOUSE IF NOT EXISTS {name} WITH"]
            sql_parts.append(f"WAREHOUSE_SIZE = '{size}'")

            if type != "STANDARD":
                sql_parts.append(f"WAREHOUSE_TYPE = '{type}'")

            sql_parts.append(f"AUTO_SUSPEND = {auto_suspend}")
            sql_parts.append(f"AUTO_RESUME = {str(auto_resume).upper()}")
            sql_parts.append(f"MIN_CLUSTER_COUNT = {min_cluster_count}")
            sql_parts.append(f"MAX_CLUSTER_COUNT = {max_cluster_count}")

            if comment:
                sql_parts.append(f"COMMENT = '{comment}'")

            sql = " ".join(sql_parts)
            self.session.sql(sql).collect()

            self.logger.info(
                f"Warehouse created: {name}", extra={"size": size, "type": type}
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to create warehouse: {name}", extra={"error": str(e)}
            )
            raise ConfigurationError(
                f"Failed to create warehouse: {name}",
                context={"warehouse": name},
                original_error=e,
            )

    def warehouse_exists(self, name: str) -> bool:
        """Check if a warehouse exists."""
        try:
            result = self.session.sql(f"SHOW WAREHOUSES LIKE '{name}'").collect()
            return len(result) > 0
        except Exception:
            return False

    def list_warehouses(self) -> List[str]:
        """List all warehouses."""
        try:
            result = self.session.sql("SHOW WAREHOUSES").collect()
            return [row["name"] for row in result]
        except Exception as e:
            self.logger.error(f"Failed to list warehouses: {e}")
            return []
