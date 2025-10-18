"""Orchestration for automated pipeline execution.

This module provides orchestration capabilities using Snowflake Tasks,
Streams, and event-driven triggers.

Classes:
    TaskOrchestrator: Manage Snowflake Tasks
    StreamProcessor: Process CDC with Streams
"""

from snowflake_ml_template.orchestration.streams import StreamProcessor
from snowflake_ml_template.orchestration.tasks import TaskOrchestrator

__all__ = ["TaskOrchestrator", "StreamProcessor"]
