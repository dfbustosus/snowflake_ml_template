"""Centralized Snowflake session management.

This module implements the Singleton pattern for managing Snowflake sessions.
It provides connection pooling, warehouse switching, and database/schema
switching capabilities.

Classes:
    SessionManager: Singleton session manager for Snowflake connections
"""

import threading
from typing import Any, Dict, Mapping, Optional, cast

from snowflake.snowpark import Session

from snowflake_ml_template.utils.logging import StructuredLogger, get_logger


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
    _SENSITIVE_CONFIG_KEYS = {
        "password",
        "private_key",
        "private_key_passphrase",
        "token",
        "oauth_token",
    }

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
        self._current_config_key: Optional[str] = None
        self._state_lock: threading.RLock = threading.RLock()
        self._logger = self._create_logger()

    def _create_logger(self) -> StructuredLogger:
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

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
        config_copy = dict(config)

        if not config_copy.get("account"):
            raise ValueError("Snowflake account is required")
        if not config_copy.get("user"):
            raise ValueError("Snowflake user is required")

        config_key = cls._get_config_hash(config_copy)

        should_apply_context = False

        with instance._state_lock:
            if config_key in instance._sessions:
                instance._logger.info(
                    f"Reusing existing session for account: {config_copy.get('account')}"
                )
                session = instance._sessions[config_key]
                instance._current_session = session
                instance._current_config = cls._sanitize_config(config_copy)
                instance._current_config_key = config_key
                should_apply_context = True
            else:
                try:
                    instance._logger.info(
                        f"Creating new Snowflake session for account: {config_copy.get('account')}"
                    )
                    session = Session.builder.configs(config_copy).create()
                except Exception as e:
                    instance._logger.error(
                        f"Failed to create Snowflake session: {str(e)}", exc_info=True
                    )
                    raise

                instance._sessions[config_key] = session
                instance._current_session = session
                instance._current_config = cls._sanitize_config(config_copy)
                instance._current_config_key = config_key

                instance._logger.info(
                    f"Successfully created session. "
                    f"Warehouse: {config_copy.get('warehouse', 'N/A')}, "
                    f"Database: {config_copy.get('database', 'N/A')}, "
                    f"Schema: {config_copy.get('schema', 'N/A')}"
                )

        if should_apply_context:
            cls._apply_context_if_needed(instance._current_session, config_copy)
        return cast(Session, instance._current_session)

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

        with instance._state_lock:
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

        if not warehouse:
            raise ValueError("Warehouse name cannot be empty")

        with instance._state_lock:
            if instance._current_session is None:
                raise RuntimeError("No active session. Call create_session() first.")

            instance._logger.info(f"Switching to warehouse: {warehouse}")

            try:
                instance._current_session.use_warehouse(warehouse)
                if instance._current_config is not None:
                    instance._current_config["warehouse"] = warehouse
                instance._logger.info(
                    f"Successfully switched to warehouse: {warehouse}"
                )
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

        if not database:
            raise ValueError("Database name cannot be empty")

        with instance._state_lock:
            if instance._current_session is None:
                raise RuntimeError("No active session. Call create_session() first.")

            instance._logger.info(
                f"Switching to database: {database}"
                + (f", schema: {schema}" if schema else "")
            )

            try:
                instance._current_session.use_database(database)
                if schema:
                    instance._current_session.use_schema(schema)

                if instance._current_config is not None:
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
        with instance._state_lock:
            if instance._current_session is None:
                instance._logger.warning("No active session to close")
                return

            session = instance._current_session
            config_key = instance._current_config_key
            instance._logger.info("Closing current session")

        try:
            session.close()
        except Exception as e:
            instance._logger.error(f"Error closing session: {str(e)}", exc_info=True)
            raise

        with instance._state_lock:
            if config_key and config_key in instance._sessions:
                del instance._sessions[config_key]
            instance._current_session = None
            instance._current_config = None
            instance._current_config_key = None
            instance._logger.info("Session closed successfully")

    @classmethod
    def close_all_sessions(cls) -> None:
        """Close all active sessions.

        This method closes all sessions in the session pool. Use this
        during application shutdown to clean up all resources.

        Example:
            >>> SessionManager.close_all_sessions()
        """
        instance = cls()

        with instance._state_lock:
            sessions = list(instance._sessions.items())
            active_count = len(sessions)
            instance._sessions.clear()
            instance._current_session = None
            instance._current_config = None
            instance._current_config_key = None

        instance._logger.info(f"Closing all sessions ({active_count} active)")

        for config_key, session in sessions:
            try:
                session.close()
            except Exception as e:
                instance._logger.error(f"Error closing session {config_key}: {str(e)}")

        instance._logger.info("All sessions closed")

    @classmethod
    def _get_config_hash(cls, config: Dict[str, Any]) -> str:
        """Generate a hash key for a configuration.

        This method creates a unique hash key based on the connection
        parameters to identify sessions.

        Args:
            config: Configuration dictionary

        Returns:
            Hash key string
        """
        # Use account, user, and role to create a unique key
        hashed_items = []
        for key in sorted(config.keys()):
            if key.lower() in cls._SENSITIVE_CONFIG_KEYS:
                continue
            value = config[key]
            hashed_items.append(f"{key}={cls._hashable_representation(value)}")
        return "|".join(hashed_items)

    @classmethod
    def _sanitize_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in config.items():
            if key.lower() in cls._SENSITIVE_CONFIG_KEYS:
                continue
            sanitized[key] = value
        return sanitized

    @classmethod
    def _hashable_representation(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(
                (k, cls._hashable_representation(v)) for k, v in sorted(value.items())
            )
        if isinstance(value, list):
            return tuple(cls._hashable_representation(v) for v in value)
        if isinstance(value, tuple):
            return tuple(cls._hashable_representation(v) for v in value)
        if isinstance(value, set):
            return tuple(
                cls._hashable_representation(v)
                for v in sorted(value, key=lambda item: repr(item))
            )
        return value

    @staticmethod
    def _needs_update(current: Optional[str], desired: Optional[str]) -> bool:
        if desired is None:
            return False
        if current is None:
            return True
        return current.upper() != desired.upper()

    @classmethod
    def _apply_context_if_needed(
        cls, session: Optional[Session], config: Dict[str, Any]
    ) -> None:
        if session is None:
            return
        role = config.get("role")
        if isinstance(role, str) and cls._needs_update(
            session.get_current_role(), role
        ):
            session.use_role(role)
        warehouse = config.get("warehouse")
        if isinstance(warehouse, str) and cls._needs_update(
            session.get_current_warehouse(), warehouse
        ):
            session.use_warehouse(warehouse)
        database = config.get("database")
        if isinstance(database, str) and cls._needs_update(
            session.get_current_database(), database
        ):
            session.use_database(database)
        schema = config.get("schema")
        if isinstance(schema, str) and cls._needs_update(
            session.get_current_schema(), schema
        ):
            session.use_schema(schema)

    @classmethod
    def get_current_warehouse(cls) -> Optional[str]:
        """Get the current warehouse name.

        Returns:
            Current warehouse name or None if no session exists
        """
        try:
            session = cls.get_session()
        except RuntimeError:
            return None
        return cast(Optional[str], session.get_current_warehouse())

    @classmethod
    def get_current_database(cls) -> Optional[str]:
        """Get the current database name.

        Returns:
            Current database name or None if no session exists
        """
        try:
            session = cls.get_session()
        except RuntimeError:
            return None
        return cast(Optional[str], session.get_current_database())

    @classmethod
    def get_current_schema(cls) -> Optional[str]:
        """Get the current schema name.

        Returns:
            Current schema name or None if no session exists
        """
        try:
            session = cls.get_session()
        except RuntimeError:
            return None
        return cast(Optional[str], session.get_current_schema())

    @classmethod
    def switch_schema(cls, schema: str) -> None:
        """Switch the active schema for the current session."""
        instance = cls()

        if not schema:
            raise ValueError("Schema name cannot be empty")

        with instance._state_lock:
            if instance._current_session is None:
                raise RuntimeError("No active session. Call create_session() first.")
            instance._logger.info(f"Switching to schema: {schema}")

            try:
                instance._current_session.use_schema(schema)
                if instance._current_config is not None:
                    instance._current_config["schema"] = schema
                instance._logger.info(f"Successfully switched to schema: {schema}")
            except Exception as e:
                instance._logger.error(
                    f"Failed to switch schema: {str(e)}", exc_info=True
                )
                raise
