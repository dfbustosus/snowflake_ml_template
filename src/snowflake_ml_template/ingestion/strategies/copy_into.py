"""COPY INTO ingestion strategy for bulk data loading."""

from typing import Any, Iterable

from snowflake_ml_template.core.base.ingestion import (
    BaseIngestionStrategy,
    CopyIntoOptions,
    DataSource,
    IngestionConfig,
    IngestionMethod,
    IngestionResult,
)


class CopyIntoStrategy(BaseIngestionStrategy):
    """Bulk data ingestion using COPY INTO command."""

    def __init__(self, config: IngestionConfig) -> None:
        """Initialize the strategy."""
        super().__init__(config)

    def _options(self) -> CopyIntoOptions:
        return self.config.copy_options or CopyIntoOptions()

    @staticmethod
    def _bool_literal(value: bool | None) -> str | None:
        if value is None:
            return None
        return "TRUE" if value else "FALSE"

    @staticmethod
    def _format_location(location: str) -> str:
        if location.startswith("@") or location.lstrip().upper().startswith("STAGE"):
            return location
        if location.lstrip().upper().startswith("SELECT"):
            return f"({location})"
        return f"'{location}'"

    def _build_copy_statement(self, source: DataSource, target: str) -> str:
        options = self._options()

        file_format_clause = (
            f"FILE_FORMAT = (FORMAT_NAME = '{options.file_format_name}')"
            if options.file_format_name
            else f"FILE_FORMAT = (TYPE = '{source.file_format}')"
        )

        clauses: list[str] = [
            f"COPY INTO {target}",
            f"FROM {self._format_location(source.location)}",
            file_format_clause,
            f"ON_ERROR = '{options.on_error or self.config.on_error}'",
            f"PURGE = {self._bool_literal(self.config.purge)}",
        ]

        if options.files:
            files_formatted = ", ".join(f"'{file}'" for file in options.files)
            clauses.append(f"FILES = ({files_formatted})")
        if options.pattern:
            clauses.append(f"PATTERN = '{options.pattern}'")
        if options.force is not None:
            clauses.append(f"FORCE = {self._bool_literal(options.force)}")
        if options.size_limit is not None:
            clauses.append(f"SIZE_LIMIT = {options.size_limit}")
        if options.max_files is not None:
            clauses.append(f"MAX_FILES = {options.max_files}")
        if options.parallel is not None:
            clauses.append(f"PARALLEL = {options.parallel}")
        if options.enable_octal is not None:
            clauses.append(f"ENABLE_OCTAL = {self._bool_literal(options.enable_octal)}")
        if options.return_failed_only:
            clauses.append("RETURN_FAILED_ONLY = TRUE")
        if options.disable_notification:
            clauses.append("DISABLE_NOTIFICATION = TRUE")
        if options.enforce_length is not None:
            clauses.append(
                f"ENFORCE_LENGTH = {self._bool_literal(options.enforce_length)}"
            )
        if options.truncate_columns is not None:
            clauses.append(
                f"TRUNCATE_COLUMNS = {self._bool_literal(options.truncate_columns)}"
            )
        if options.trim_space is not None:
            clauses.append(f"TRIM_SPACE = {self._bool_literal(options.trim_space)}")
        if options.null_if:
            null_literals = ", ".join(f"'{value}'" for value in options.null_if)
            clauses.append(f"NULL_IF = ({null_literals})")

        return "\n".join(clauses)

    @staticmethod
    def _rows_loaded(result_rows: Iterable[Any]) -> tuple[int, int]:
        rows_loaded = 0
        files_processed = 0
        for row in result_rows:
            files_processed += 1
            if isinstance(row, dict):
                value = row.get("rows_loaded")
            elif hasattr(row, "get"):
                get_method = getattr(row, "get")
                value = get_method("rows_loaded")
            else:
                value = getattr(row, "rows_loaded", None)
            rows_loaded += int(value or 0)
        return rows_loaded, files_processed

    def ingest(self, source: DataSource, target: str, **kwargs: Any) -> IngestionResult:
        """Ingest data using COPY INTO command."""
        options = self._options()

        self.session.sql(f"USE WAREHOUSE {self.config.warehouse}").collect()

        statement = self._build_copy_statement(source, target)
        result_rows = self.session.sql(statement).collect()
        rows_loaded, files_processed = self._rows_loaded(result_rows)

        return IngestionResult(
            status="success",
            method=IngestionMethod.COPY_INTO,
            target_table=target,
            rows_loaded=rows_loaded,
            files_processed=files_processed,
            metadata={
                "copy_options": {
                    "files": list(options.files or []),
                    "pattern": options.pattern,
                    "file_format_name": options.file_format_name,
                    "force": options.force,
                    "size_limit": options.size_limit,
                    "parallel": options.parallel,
                    "enforce_length": options.enforce_length,
                    "truncate_columns": options.truncate_columns,
                    "trim_space": options.trim_space,
                    "null_if": list(options.null_if or []),
                }
            },
        )

    def validate(self) -> bool:
        """Validate strategy configuration."""
        options = self._options()
        if options.files and options.pattern:
            return False
        return bool(self.config.source.location and self.config.target_table)
