"""Centralized Snowflake session management.

This module implements the Singleton pattern for managing Snowflake sessions.
It provides connection pooling, warehouse switching, and database/schema
switching capabilities.

Classes:
    SessionManager: Singleton session manager for Snowflake connections
"""

import threading
from typing import Any, Dict, Optional

from snowflake.snowpark import Session


class SessionManager:
    """Singleton session manager for Snowflake connections.

    This class manages Snowflake sessions using the Singleton pattern to ensure
    only one session instance exists per configuration. It provides methods for:
    - Creating and retrieving sessions
    - Switching warehouses
    - Switching databases and schemas
    - Closing sessions

    The manager is thread-safe and can be used in multi-threaded environments.

    Attributes:
        _instance: Singleton instance
        _lock: Thread lock for thread-safe singleton creation
        _sessions: Dictionary of active sessions keyed by connection params hash
        _current_session: Currently active session

    Example:
        >>> # Create a session
        >>> config = {
        ...     "account": "my_account",
        ...     "user": "my_user",
        ...     "password": "my_password",
        ...     "warehouse": "ML_TRAINING_WH",
        ...     "database": "ML_DEV_DB",
        ...     "schema": "FEATURES"
        ... }
        >>> session = SessionManager.create_session(config)
        >>>
        >>> # Get the current session
        >>> session = SessionManager.get_session()
        >>>
        >>> # Switch warehouse
        >>> SessionManager.switch_warehouse("INFERENCE_WH")
        >>>
        >>> # Switch database and schema
        >>> SessionManager.switch_database("ML_PROD_DB", "MODELS")
    """

    _instance: Optional["SessionManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SessionManager":
        """Create or return the singleton instance.

        This method implements the Singleton pattern with thread-safety.

        Returns:
            The singleton SessionManager instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the session manager.

        This method is called once when the singleton is first created.
        """
        self._sessions: Dict[str, Session] = {}
        self._current_session: Optional[Session] = None
        self._current_config: Optional[Dict[str, Any]] = None
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

    @classmethod
    def create_session(cls, config: Dict[str, Any]) -> Session:
        """Create a new Snowflake session.

        This method creates a new Snowflake session with the provided configuration.
        If a session with the same configuration already exists, it returns the
        existing session instead of creating a new one.

        Args:
            config: Dictionary containing Snowflake connection parameters.
                Required keys: account, user
                Optional keys: password, authenticator, warehouse, database,
                              schema, role, and other Snowpark Session.builder parameters

        Returns:
            Active Snowflake session

        Raises:
            ValueError: If required configuration parameters are missing
            Exception: If session creation fails

        Example:
            >>> config = {
            ...     "account": "my_account",
            ...     "user": "my_user",
            ...     "password": "my_password",
            ...     "warehouse": "ML_TRAINING_WH"
            ... }
            >>> session = SessionManager.create_session(config)
        """
        instance = cls()

        # Validate required parameters
        if not config.get("account"):
            raise ValueError("Snowflake account is required")
        if not config.get("user"):
            raise ValueError("Snowflake user is required")

        # Create a hash key for this configuration
        config_key = cls._get_config_hash(config)

        # Check if session already exists
        if config_key in instance._sessions:
            instance._logger.info(
                f"Reusing existing session for account: {config.get('account')}"
            )
            instance._current_session = instance._sessions[config_key]
            instance._current_config = config
            return instance._current_session

        # Create new session
        try:
            instance._logger.info(
                f"Creating new Snowflake session for account: {config.get('account')}"
            )

            # Build session using Snowpark Session.builder
            builder = Session.builder.configs(config)
            session = builder.create()

            # Store session
            instance._sessions[config_key] = session
            instance._current_session = session
            instance._current_config = config

            instance._logger.info(
                f"Successfully created session. "
                f"Warehouse: {config.get('warehouse', 'N/A')}, "
                f"Database: {config.get('database', 'N/A')}, "
                f"Schema: {config.get('schema', 'N/A')}"
            )

            return session

        except Exception as e:
            instance._logger.error(
                f"Failed to create Snowflake session: {str(e)}", exc_info=True
            )
            raise

    @classmethod
    def get_session(cls) -> Session:
        """Get the current active session.

        Returns:
            Current active Snowflake session

        Raises:
            RuntimeError: If no session has been created

        Example:
            >>> session = SessionManager.get_session()
        """
        instance = cls()

        if instance._current_session is None:
            raise RuntimeError("No active session. Call create_session() first.")

        return instance._current_session

    @classmethod
    def switch_warehouse(cls, warehouse: str) -> None:
        """Switch to a different warehouse.

        This method switches the current session to use a different warehouse.
        This is useful for optimizing costs by using different warehouse sizes
        for different operations (e.g., small for ingestion, large for training).

        Args:
            warehouse: Name of the warehouse to switch to

        Raises:
            RuntimeError: If no active session exists
            ValueError: If warehouse name is empty

        Example:
            >>> SessionManager.switch_warehouse("ML_TRAINING_WH")
        """
        instance = cls()

        if instance._current_session is None:
            raise RuntimeError("No active session. Call create_session() first.")

        if not warehouse:
            raise ValueError("Warehouse name cannot be empty")

        instance._logger.info(f"Switching to warehouse: {warehouse}")

        try:
            instance._current_session.use_warehouse(warehouse)

            # Update current config
            if instance._current_config:
                instance._current_config["warehouse"] = warehouse

            instance._logger.info(f"Successfully switched to warehouse: {warehouse}")

        except Exception as e:
            instance._logger.error(
                f"Failed to switch warehouse: {str(e)}", exc_info=True
            )
            raise

    @classmethod
    def switch_database(cls, database: str, schema: Optional[str] = None) -> None:
        """Switch to a different database and optionally schema.

        This method switches the current session to use a different database
        and schema. This is useful for accessing data in different environments
        (dev, test, prod) or different schemas within the same database.

        Args:
            database: Name of the database to switch to
            schema: Optional name of the schema to switch to

        Raises:
            RuntimeError: If no active session exists
            ValueError: If database name is empty

        Example:
            >>> SessionManager.switch_database("ML_PROD_DB", "FEATURES")
        """
        instance = cls()

        if instance._current_session is None:
            raise RuntimeError("No active session. Call create_session() first.")

        if not database:
            raise ValueError("Database name cannot be empty")

        instance._logger.info(
            f"Switching to database: {database}"
            + (f", schema: {schema}" if schema else "")
        )

        try:
            instance._current_session.use_database(database)

            if schema:
                instance._current_session.use_schema(schema)

            # Update current config
            if instance._current_config:
                instance._current_config["database"] = database
                if schema:
                    instance._current_config["schema"] = schema

            instance._logger.info(
                f"Successfully switched to database: {database}"
                + (f", schema: {schema}" if schema else "")
            )

        except Exception as e:
            instance._logger.error(
                f"Failed to switch database/schema: {str(e)}", exc_info=True
            )
            raise

    @classmethod
    def close_session(cls) -> None:
        """Close the current active session.

        This method closes the current session and removes it from the
        session pool. Use this when you're done with a session to free
        up resources.

        Example:
            >>> SessionManager.close_session()
        """
        instance = cls()

        if instance._current_session is None:
            instance._logger.warning("No active session to close")
            return

        try:
            instance._logger.info("Closing current session")
            instance._current_session.close()

            # Remove from sessions dict
            if instance._current_config:
                config_key = cls._get_config_hash(instance._current_config)
                if config_key in instance._sessions:
                    del instance._sessions[config_key]

            instance._current_session = None
            instance._current_config = None

            instance._logger.info("Session closed successfully")

        except Exception as e:
            instance._logger.error(f"Error closing session: {str(e)}", exc_info=True)
            raise

    @classmethod
    def close_all_sessions(cls) -> None:
        """Close all active sessions.

        This method closes all sessions in the session pool. Use this
        during application shutdown to clean up all resources.

        Example:
            >>> SessionManager.close_all_sessions()
        """
        instance = cls()

        instance._logger.info(
            f"Closing all sessions ({len(instance._sessions)} active)"
        )

        for config_key, session in list(instance._sessions.items()):
            try:
                session.close()
                del instance._sessions[config_key]
            except Exception as e:
                instance._logger.error(f"Error closing session {config_key}: {str(e)}")

        instance._current_session = None
        instance._current_config = None

        instance._logger.info("All sessions closed")

    @staticmethod
    def _get_config_hash(config: Dict[str, Any]) -> str:
        """Generate a hash key for a configuration.

        This method creates a unique hash key based on the connection
        parameters to identify sessions.

        Args:
            config: Configuration dictionary

        Returns:
            Hash key string
        """
        # Use account, user, and role to create a unique key
        key_parts = [
            config.get("account", ""),
            config.get("user", ""),
            config.get("role", ""),
        ]
        return "|".join(key_parts)

    @classmethod
    def get_current_warehouse(cls) -> Optional[str]:
        """Get the current warehouse name.

        Returns:
            Current warehouse name or None if no session exists
        """
        instance = cls()
        if instance._current_config:
            return instance._current_config.get("warehouse")
        return None

    @classmethod
    def get_current_database(cls) -> Optional[str]:
        """Get the current database name.

        Returns:
            Current database name or None if no session exists
        """
        instance = cls()
        if instance._current_config:
            return instance._current_config.get("database")
        return None

    @classmethod
    def get_current_schema(cls) -> Optional[str]:
        """Get the current schema name.

        Returns:
            Current schema name or None if no session exists
        """
        instance = cls()
        if instance._current_config:
            return instance._current_config.get("schema")
        return None
