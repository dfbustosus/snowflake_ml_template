"""Data ingestion engine with multiple strategies.

This module provides data ingestion capabilities using the Strategy pattern,
supporting multiple ingestion methods (Snowpipe, COPY INTO, Streaming).

Classes:
    IngestionOrchestrator: Orchestrate ingestion operations
"""

from snowflake_ml_template.ingestion.orchestrator import IngestionOrchestrator
from snowflake_ml_template.ingestion.strategies.copy_into import CopyIntoStrategy
from snowflake_ml_template.ingestion.strategies.snowpipe import SnowpipeStrategy

__all__ = ["IngestionOrchestrator", "SnowpipeStrategy", "CopyIntoStrategy"]
