"""Snowpipe ingestion strategy for continuous data loading."""

from datetime import datetime
from typing import Any, Dict

from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    DataSource,
    IngestionConfig,
    IngestionMethod,
    IngestionResult,
    SnowpipeLoadMethod,
    SnowpipeOptions,
)
from snowflake_ml_template.ingestion.streaming import SnowpipeStreamingClient


class SnowpipeStrategy(BaseIngestionStrategy):
    """Continuous data ingestion using Snowpipe."""

    def __init__(self, config: IngestionConfig) -> None:
        """Initialize the strategy."""
        super().__init__(config)

    def _options(self) -> SnowpipeOptions:
        return self.config.snowpipe_options or SnowpipeOptions()

    def _pipe_name(self, target: str) -> str:
        options = self._options()
        if options.pipe_name:
            return options.pipe_name
        sanitized = target.replace(".", "_").replace('"', "")
        return f"{sanitized}_PIPE"

    def _build_copy_clause(self, source: DataSource, target: str) -> str:
        options = self._options()
        file_format_clause = (
            f"FILE_FORMAT => (FORMAT_NAME => '{options.file_format_name}')"
            if options.file_format_name
            else f"FILE_FORMAT => (TYPE => '{source.file_format}')"
        )

        clauses = [
            f"COPY INTO {target}",
            f"FROM '{source.location}'",
            file_format_clause,
            f"ON_ERROR => '{self.config.on_error}'",
        ]

        if options.pattern:
            clauses.append(f"PATTERN => '{options.pattern}'")

        return "\n".join(clauses)

    def _create_or_replace_pipe(self, pipe_name: str, copy_clause: str) -> None:
        options = self._options()
        header_parts = [f"CREATE OR REPLACE PIPE {pipe_name}"]
        if options.auto_ingest:
            header_parts.append("AUTO_INGEST = TRUE")
        if options.integration:
            header_parts.append(f"INTEGRATION = {options.integration}")
        if options.notification_channel:
            channel = options.notification_channel
            header_parts.append(f"NOTIFICATION_CHANNEL = '{channel.channel_arn}'")
        if options.error_integration:
            header_parts.append(f"ERROR_INTEGRATION = '{options.error_integration}'")
        if options.comment:
            header_parts.append(f"COMMENT = '{options.comment}'")

        header = " ".join(header_parts)
        statement = f"{header}\nAS\n{copy_clause}"
        self.session.sql(statement).collect()

    def _refresh_pipe(self, pipe_name: str, **kwargs: Any) -> None:
        prefix = kwargs.get("refresh_prefix")
        modified_after = kwargs.get("modified_after")
        if prefix or modified_after:
            modifiers = []
            if prefix:
                modifiers.append(f"PREFIX = '{prefix}'")
            if modified_after:
                if isinstance(modified_after, datetime):
                    modified_after = modified_after.strftime("%Y-%m-%d %H:%M:%S")
                modifiers.append(f"MODIFIED_AFTER = TO_TIMESTAMP('{modified_after}')")
            modifier_sql = " ".join(modifiers)
            self.session.sql(f"ALTER PIPE {pipe_name} REFRESH {modifier_sql}").collect()
        else:
            self.session.sql(f"ALTER PIPE {pipe_name} REFRESH").collect()

    def _trigger_rest_refresh(self, pipe_name: str, **kwargs: Any) -> None:
        files = kwargs.get("files")
        if files:
            for file_name in files:
                self.session.sql(
                    f"CALL SYSTEM$PIPE_FORCE_REFRESH('{pipe_name}', '{file_name}')"
                ).collect()
        else:
            self._refresh_pipe(pipe_name, **kwargs)

    def _capture_pipe_status(self, pipe_name: str) -> Dict[str, Any]:
        try:
            rows = self.session.sql(
                f"SELECT SYSTEM$PIPE_STATUS('{pipe_name}') AS STATUS"
            ).collect()
        except Exception:
            return {}
        if not rows:
            return {}
        status_variant = rows[0].get("STATUS") or rows[0].get("status")
        if isinstance(status_variant, dict):
            return status_variant
        return {"raw_status": status_variant}

    def ingest(self, source: DataSource, target: str, **kwargs: Any) -> IngestionResult:
        """Ingest data using Snowpipe."""
        options = self._options()
        pipe_name = self._pipe_name(target)

        copy_clause = self._build_copy_clause(source, target)
        self._create_or_replace_pipe(pipe_name, copy_clause)

        metadata: Dict[str, Any] = {"pipe_name": pipe_name}

        if options.load_method == SnowpipeLoadMethod.REST_API:
            self._trigger_rest_refresh(pipe_name, **kwargs)
            metadata["rest_refresh"] = {
                "files": kwargs.get("files"),
                "prefix": kwargs.get("refresh_prefix"),
            }
        elif options.load_method == SnowpipeLoadMethod.STREAMING:
            rows = kwargs.get("rows")
            if rows is None:
                raise ValueError("rows must be provided for Snowpipe streaming")
            streaming_client = kwargs.get("streaming_client")
            if streaming_client is None:
                streaming_client = SnowpipeStreamingClient(
                    session=self.session,
                    pipe_name=pipe_name,
                    warehouse=options.warehouse,
                    role=options.role,
                )
            batch_id = streaming_client.insert_rows(target, rows)
            metadata["streaming"] = {"batch_id": batch_id, "row_count": len(rows)}
        elif options.auto_ingest and kwargs.get("refresh_on_create"):
            self._refresh_pipe(pipe_name, **kwargs)
            metadata["auto_ingest_refresh"] = True

        status_snapshot = self._capture_pipe_status(pipe_name)
        if status_snapshot:
            metadata["pipe_status"] = status_snapshot

        return IngestionResult(
            status="success",
            method=IngestionMethod.SNOWPIPE,
            target_table=target,
            metadata=metadata,
        )

    def validate(self) -> bool:
        """Validate strategy configuration."""
        options = self._options()
        if not self.config or not self.config.source:
            return False
        if not self.config.source.location:
            return False
        if not self.config.target_table:
            return False
        if options.auto_ingest and not (
            options.integration or options.notification_channel
        ):
            return False
        if options.load_method == SnowpipeLoadMethod.REST_API and not options.warehouse:
            return False
        if options.load_method == SnowpipeLoadMethod.STREAMING and not options.role:
            return False
        return True

    def get_target_table_name(self) -> str:
        """Get target table name."""
        if not self.config or not self.config.target_table:
            raise ValueError("Target table is not configured")
        return self.config.target_table
