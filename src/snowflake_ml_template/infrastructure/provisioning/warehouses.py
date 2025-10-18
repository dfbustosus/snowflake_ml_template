"""Warehouse provisioning for Snowflake MLOps.

This module provides functionality to create and manage Snowflake warehouses
optimized for different ML workload types.

Classes:
    WarehouseProvisioner: Create and manage warehouses
"""

from typing import Dict, Iterable, List, Optional, TypedDict

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.tracking import ExecutionEventTracker
from snowflake_ml_template.infrastructure.provisioning.base import BaseProvisioner


class WarehouseConfig(TypedDict):
    """Type definition for warehouse configuration."""

    size: str
    type: str
    auto_suspend: int
    min_cluster_count: int
    max_cluster_count: int
    comment: str


class WarehouseProvisioner(BaseProvisioner):
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

    def __init__(
        self,
        session: Session,
        tracker: Optional[ExecutionEventTracker] = None,
    ) -> None:
        """Initialize `WarehouseProvisioner` with a Snowflake session."""
        super().__init__(session=session, tracker=tracker)

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
        *,
        statement_timeout_in_seconds: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        resource_monitor: Optional[str] = None,
        scaling_policy: Optional[str] = None,
        grants_to_roles: Optional[Iterable[str]] = None,
    ) -> bool:
        """Create a warehouse if it doesn't exist."""
        if not name:
            raise ValueError("Warehouse name cannot be empty")

        warehouse_identifier = self.quote_identifier(name)
        self.logger.info(
            "Creating warehouse",
            extra={
                "warehouse": name,
                "size": size,
                "type": type,
            },
        )

        clauses = [f"CREATE WAREHOUSE IF NOT EXISTS {warehouse_identifier} WITH"]
        clauses.append(f"WAREHOUSE_SIZE = {self.quote_literal(size)}")

        if type != "STANDARD":
            clauses.append(f"WAREHOUSE_TYPE = {self.quote_literal(type)}")

        clauses.append(f"AUTO_SUSPEND = {auto_suspend}")
        clauses.append(f"AUTO_RESUME = {str(auto_resume).upper()}")
        clauses.append(f"MIN_CLUSTER_COUNT = {min_cluster_count}")
        clauses.append(f"MAX_CLUSTER_COUNT = {max_cluster_count}")

        if statement_timeout_in_seconds is not None:
            clauses.append(
                f"STATEMENT_TIMEOUT_IN_SECONDS = {statement_timeout_in_seconds}"
            )

        if scaling_policy:
            clauses.append(f"SCALING_POLICY = {self.quote_literal(scaling_policy)}")

        if resource_monitor:
            clauses.append(
                f"RESOURCE_MONITOR = {self.quote_identifier(resource_monitor)}"
            )

        if comment:
            clauses.append(f"COMMENT = {self.quote_literal(comment)}")

        sql = " ".join(clauses)

        with self.transactional():
            self._execute_sql(
                sql,
                context={"warehouse": name},
                emit_event="warehouse_created",
            )

            if tags:
                self._apply_tags("WAREHOUSE", warehouse_identifier, tags)

            if grants_to_roles:
                for role in grants_to_roles:
                    grant_sql = (
                        f"GRANT OWNERSHIP ON WAREHOUSE {warehouse_identifier} "
                        f"TO ROLE {self.quote_identifier(role)}"
                    )
                    self._execute_sql(
                        grant_sql,
                        context={"warehouse": name, "role": role},
                        emit_event="warehouse_grant_ownership",
                    )

        return True

    def warehouse_exists(self, name: str) -> bool:
        """Check if a warehouse exists."""
        try:
            result = self.session.sql(
                f"SHOW WAREHOUSES LIKE {self.quote_literal(name)}"
            ).collect()
            return len(result) > 0
        except Exception:
            return False

    def list_warehouses(self) -> List[str]:
        """List all warehouses."""
        try:
            result = self.session.sql("SHOW WAREHOUSES").collect()
            return [row["name"] for row in result]
        except Exception as e:
            self.logger.error("Failed to list warehouses", extra={"error": str(e)})
            return []
