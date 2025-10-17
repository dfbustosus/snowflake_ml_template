"""Context manager for temporary Snowflake session changes.

This module provides a context manager for temporarily changing session
settings (warehouse, database, schema) with automatic rollback.

Classes:
    SessionContext: Context manager for temporary session changes
"""

from typing import Any, Literal, Optional

from snowflake_ml_template.core.session.manager import SessionManager


class SessionContext:
    """Context manager for temporary session changes.

    This class provides a context manager that temporarily changes session
    settings (warehouse, database, schema) and automatically restores the
    previous settings when exiting the context, even if an exception occurs.

    This is useful for operations that need to temporarily use different
    resources without affecting the global session state.

    Attributes:
        warehouse: Warehouse to use in this context
        database: Database to use in this context
        schema: Schema to use in this context
        _previous_warehouse: Previous warehouse (for rollback)
        _previous_database: Previous database (for rollback)
        _previous_schema: Previous schema (for rollback)

    Example:
        >>> # Temporarily switch to a different warehouse
        >>> with SessionContext(warehouse="INFERENCE_WH"):
        ...     # Code here uses INFERENCE_WH
        ...     result = session.sql("SELECT ...").collect()
        >>> # Automatically switched back to previous warehouse
        >>>
        >>> # Temporarily switch database and schema
        >>> with SessionContext(database="ML_PROD_DB", schema="MODELS"):
        ...     # Code here uses ML_PROD_DB.MODELS
        ...     model = session.table("my_model")
        >>> # Automatically switched back
        >>>
        >>> # Nested contexts are supported
        >>> with SessionContext(warehouse="TRANSFORM_WH"):
        ...     # Use TRANSFORM_WH
        ...     with SessionContext(warehouse="ML_TRAINING_WH"):
        ...         # Use ML_TRAINING_WH
        ...         pass
        ...     # Back to TRANSFORM_WH
        >>> # Back to original warehouse
    """

    def __init__(
        self,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        """Initialize the session context.

        Args:
            warehouse: Optional warehouse to use in this context
            database: Optional database to use in this context
            schema: Optional schema to use in this context

        Raises:
            ValueError: If schema is specified without database
        """
        if schema and not database:
            raise ValueError("Schema cannot be specified without database")

        self.warehouse = warehouse
        self.database = database
        self.schema = schema

        self._previous_warehouse: Optional[str] = None
        self._previous_database: Optional[str] = None
        self._previous_schema: Optional[str] = None

        self._logger = self._get_logger()

    def _get_logger(self) -> Any:
        """Get logger instance.

        This is a placeholder that will be replaced with proper
        structured logging in Day 3.

        Returns:
            Logger instance
        """
        import logging

        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def __enter__(self) -> "SessionContext":
        """Enter the context manager.

        This method saves the current session settings and applies the
        new settings specified in the constructor.

        Returns:
            Self for use in 'with' statement

        Raises:
            RuntimeError: If no active session exists
        """
        self._logger.debug("Entering SessionContext")

        # Save current settings
        self._previous_warehouse = SessionManager.get_current_warehouse()
        self._previous_database = SessionManager.get_current_database()
        self._previous_schema = SessionManager.get_current_schema()

        # Apply new settings
        if self.warehouse:
            self._logger.debug(f"Switching to warehouse: {self.warehouse}")
            SessionManager.switch_warehouse(self.warehouse)

        if self.database:
            self._logger.debug(
                f"Switching to database: {self.database}"
                + (f", schema: {self.schema}" if self.schema else "")
            )
            SessionManager.switch_database(self.database, self.schema)

        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> Literal[False]:
        """Exit the context manager.

        This method restores the previous session settings, even if an
        exception occurred in the context.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred

        Returns:
            False to propagate any exception that occurred
        """
        self._logger.debug("Exiting SessionContext")

        try:
            # Restore previous settings in reverse order
            if self.database and self._previous_database:
                self._logger.debug(
                    f"Restoring database: {self._previous_database}"
                    + (
                        f", schema: {self._previous_schema}"
                        if self._previous_schema
                        else ""
                    )
                )
                SessionManager.switch_database(
                    self._previous_database, self._previous_schema
                )

            if self.warehouse and self._previous_warehouse:
                self._logger.debug(f"Restoring warehouse: {self._previous_warehouse}")
                SessionManager.switch_warehouse(self._previous_warehouse)

        except Exception as e:
            self._logger.error(
                f"Error restoring session settings: {str(e)}", exc_info=True
            )
            # Don't suppress the original exception

        # Return False to propagate any exception that occurred in the context
        return False

    def __repr__(self) -> str:
        """Get string representation of the context.

        Returns:
            String representation
        """
        parts = []
        if self.warehouse:
            parts.append(f"warehouse={self.warehouse}")
        if self.database:
            parts.append(f"database={self.database}")
        if self.schema:
            parts.append(f"schema={self.schema}")

        return f"SessionContext({', '.join(parts)})"
