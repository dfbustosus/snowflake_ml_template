"""Tracking interfaces for base components.

This module defines lightweight protocols that allow pipeline components to
emit structured events without taking hard dependencies on specific tracking
backends. Implementations can forward events to Snowflake tables, external
telemetry systems, or any custom observability tooling.
"""

from __future__ import annotations

import logging
from typing import Dict, Protocol


class ExecutionEventTracker(Protocol):
    """Protocol for receiving structured execution events.

    Implementations should be resilient and avoid raising exceptions. The
    payload is intentionally unstructured to give flexibility to downstream
    consumers.
    """

    def record_event(
        self, component: str, event: str, payload: Dict[str, object]
    ) -> None:
        """Record an execution event for the specified component."""
        ...


def emit_tracker_event(
    tracker: ExecutionEventTracker | None,
    component: str,
    event: str,
    payload: Dict[str, object],
) -> None:
    """Safely emit an event through the tracker if one is configured."""
    if tracker is None:
        return

    try:
        tracker.record_event(component=component, event=event, payload=payload)
    except Exception:  # pragma: no cover - defensive logging hook
        # Trackers should not break pipeline execution; failures are ignored.
        logging.getLogger(__name__).exception("Tracker event emission failed")
