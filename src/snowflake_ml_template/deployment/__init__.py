"""Model deployment engine with multiple strategies.

This module provides model deployment capabilities using the Strategy pattern,
supporting Warehouse UDF and SPCS deployment methods.

Classes:
    DeploymentOrchestrator: Orchestrate deployment operations
    WarehouseUDFStrategy: Deploy as Warehouse UDF
    SPCSStrategy: Deploy to Snowpark Container Services
"""

from snowflake_ml_template.deployment.orchestrator import DeploymentOrchestrator
from snowflake_ml_template.deployment.strategies.spcs import SPCSStrategy
from snowflake_ml_template.deployment.strategies.warehouse_udf import (
    WarehouseUDFStrategy,
)

__all__ = ["DeploymentOrchestrator", "WarehouseUDFStrategy", "SPCSStrategy"]
