"""Helpers for Snowpipe Streaming ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence
from uuid import uuid4

from snowflake_ml_template.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from snowflake.snowpark import Session
else:  # pragma: no cover - runtime import with fallback
    try:
        from snowflake.snowpark import Session
    except ImportError:  # pragma: no cover - handled gracefully in tests
        Session = Any  # type: ignore[assignment]


logger = get_logger(__name__)


class SnowpipeStreamingClient:
    """Client utility for Snowpipe Streaming inserts.

    This helper batches a list of dictionaries into a Snowpark DataFrame and
    appends the result to the target table. It is intentionally lightweight so
    that it can be swapped out for a more advanced streaming connector when
    required by production workloads.
    """

    def __init__(
        self,
        session: Session,
        pipe_name: str,
        *,
        warehouse: str | None = None,
        role: str | None = None,
    ) -> None:
        """Initialize the client."""
        if session is None:
            raise ValueError("session cannot be None")
        if not pipe_name:
            raise ValueError("pipe_name must be provided")

        self._session = session
        self._pipe_name = pipe_name
        self._warehouse = warehouse
        self._role = role

    def _ensure_context(self) -> None:
        """Ensure the Snowflake session uses the expected role and warehouse."""
        if self._role:
            self._session.sql(f"USE ROLE {self._role}").collect()
        if self._warehouse:
            self._session.sql(f"USE WAREHOUSE {self._warehouse}").collect()

    def insert_rows(
        self,
        target_table: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> str:
        """Insert rows into the target table using Snowpark."""
        if not rows:
            raise ValueError("rows must contain at least one record")

        self._ensure_context()

        # Normalise column ordering for deterministic inserts
        first_row = rows[0]
        if not isinstance(first_row, Mapping):
            raise TypeError("rows must be a sequence of mapping objects")

        columns = list(first_row.keys())
        for row in rows:
            missing = set(columns) - set(row.keys())
            if missing:
                raise ValueError(f"Row {row} is missing columns: {sorted(missing)}")

        data = [[row[col] for col in columns] for row in rows]
        dataframe = self._session.create_dataframe(data, schema=columns)
        dataframe.write.mode("append").save_as_table(target_table)

        batch_id = f"{self._pipe_name}-{uuid4().hex}"
        logger.info(
            "Snowpipe streaming batch inserted",
            extra={
                "pipe_name": self._pipe_name,
                "target_table": target_table,
                "row_count": len(rows),
                "batch_id": batch_id,
            },
        )
        return batch_id
