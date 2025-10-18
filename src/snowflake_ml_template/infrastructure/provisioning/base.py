"""Shared tooling for Snowflake infrastructure provisioners."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.tracking import (
    ExecutionEventTracker,
    emit_tracker_event,
)
from snowflake_ml_template.core.exceptions import ConfigurationError
from snowflake_ml_template.utils.logging import StructuredLogger, get_logger


@dataclass(frozen=True)
class SqlExecutionResult:
    """Envelope for Snowflake SQL execution metadata."""

    query_id: Optional[str]
    rows: Sequence[Any]


class BaseProvisioner:
    """Provide shared safeguards for Snowflake provisioning helpers."""

    def __init__(
        self,
        session: Session,
        tracker: Optional[ExecutionEventTracker] = None,
    ) -> None:
        """Initialize `BaseProvisioner` with a Snowflake session."""
        if session is None:
            raise ValueError("Session cannot be None")
        self.session = session
        self.logger: StructuredLogger = get_logger(self.__class__.__module__)
        self._tracker = tracker

    def set_tracker(self, tracker: Optional[ExecutionEventTracker]) -> None:
        """Attach an execution tracker for telemetry hooks."""
        self._tracker = tracker

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------
    @staticmethod
    def quote_identifier(value: str) -> str:
        """Safely quote a Snowflake identifier."""
        if not value:
            raise ValueError("Identifier cannot be empty")
        escaped = value.replace('"', '""')
        return f'"{escaped}"'

    @classmethod
    def format_qualified_identifier(cls, *parts: str) -> str:
        """Format a qualified identifier quoting each part."""
        if not parts:
            raise ValueError("Qualified identifier requires at least one part")
        return ".".join(cls.quote_identifier(part) for part in parts if part)

    @staticmethod
    def quote_literal(value: Optional[str]) -> Optional[str]:
        """Safely quote a string literal for Snowflake SQL."""
        if value is None:
            return None
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    def _emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        if not self._tracker:
            return
        emit_tracker_event(
            tracker=self._tracker,
            component=self.__class__.__name__,
            event=event,
            payload=payload,
        )

    @staticmethod
    def _format_set_options(options: Dict[str, Any]) -> str:
        clauses: list[str] = []
        for key, value in options.items():
            if value is None:
                continue
            literal: str
            if isinstance(value, bool):
                literal = str(value).upper()
            elif isinstance(value, (int, float)):
                literal = str(value)
            else:
                literal = BaseProvisioner.quote_literal(str(value)) or "NULL"
            clauses.append(f"{key} = {literal}")
        return " ".join(clauses)

    def _apply_tags(
        self,
        object_type: str,
        qualified_name: str,
        tags: Optional[Dict[str, str]],
    ) -> None:
        if not tags:
            return
        for tag_name, tag_value in tags.items():
            if not tag_name:
                raise ValueError("Tag name cannot be empty")
            tag_parts = [part for part in tag_name.split(".") if part]
            tag_qualified = self.format_qualified_identifier(*tag_parts)
            literal = self.quote_literal(tag_value)
            sql = (
                f"ALTER {object_type} {qualified_name} "
                f"SET TAG {tag_qualified} = {literal}"
            )
            self._execute_sql(sql, context={"qualified_name": qualified_name})

    def _execute_sql(
        self,
        sql: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        emit_event: Optional[str] = None,
    ) -> SqlExecutionResult:
        """Execute SQL capturing query id and wrapping ConfigurationError."""
        self.logger.debug("Executing SQL", extra={"sql": sql, "context": context})
        try:
            dataframe = self.session.sql(sql)
            rows = dataframe.collect()
            query_id = getattr(dataframe, "_statement", None)
            if query_id is None:
                query_id = getattr(dataframe, "_cursor", None)
                if query_id is not None:
                    query_id = getattr(query_id, "sfqid", None)
            if emit_event:
                event_payload = {"sql": sql}
                if context:
                    event_payload.update(context)
                if query_id:
                    event_payload["query_id"] = query_id
                self._emit_event(emit_event, event_payload)
            return SqlExecutionResult(query_id=query_id, rows=rows)
        except Exception as exc:  # pragma: no cover - Snowflake errors mocked in tests
            error_context = {"sql": sql}
            if context:
                error_context.update(context)
            query_id = getattr(exc, "sfqid", None)
            if query_id:
                error_context["query_id"] = query_id
            raise ConfigurationError(
                "Snowflake execution failed",
                context=error_context,
                original_error=exc,
            ) from exc

    @contextmanager
    def transactional(
        self, *, rollback: Optional[Iterable[str]] = None
    ) -> Iterator[None]:
        """Best-effort transactional scope executing rollback statements on error."""
        try:
            yield
        except Exception:
            if rollback:
                for statement in rollback:
                    try:
                        self.session.sql(statement).collect()
                    except Exception:
                        self.logger.warning(
                            "Rollback statement failed",
                            extra={"statement": statement},
                        )
            raise
