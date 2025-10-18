"""Unit tests for SessionManager and SessionContext.

These tests validate session management without requiring a Snowflake connection.
They use mocks to simulate Snowflake sessions.
"""

from unittest.mock import Mock, patch

import pytest

from snowflake_ml_template.core.session.context import SessionContext
from snowflake_ml_template.core.session.manager import SessionManager


class TestSessionManager:
    """Test cases for SessionManager class."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        SessionManager._instance = None

    def test_singleton_pattern(self) -> None:
        """Test that SessionManager implements singleton pattern."""
        manager1 = SessionManager()
        manager2 = SessionManager()

        assert manager1 is manager2

    @patch("snowflake_ml_template.core.session.manager.Session")
    def test_create_session_success(self, mock_session_class: Mock) -> None:
        """Test successful session creation."""
        # Setup mock
        mock_session = Mock()
        mock_builder = Mock()
        mock_builder.configs.return_value = mock_builder
        mock_builder.create.return_value = mock_session
        mock_session_class.builder = mock_builder

        # Create session
        config = {
            "account": "test_account",
            "user": "test_user",
            "password": "test_password",
            "warehouse": "TEST_WH",
        }

        session = SessionManager.create_session(config)

        assert session == mock_session
        mock_builder.configs.assert_called_once_with(config)
        mock_builder.create.assert_called_once()

    def test_create_session_missing_account_raises_error(self) -> None:
        """Test that missing account raises ValueError."""
        config = {"user": "test_user", "password": "test_password"}

        with pytest.raises(ValueError, match="Snowflake account is required"):
            SessionManager.create_session(config)

    def test_create_session_missing_user_raises_error(self) -> None:
        """Test that missing user raises ValueError."""
        config = {"account": "test_account", "password": "test_password"}

        with pytest.raises(ValueError, match="Snowflake user is required"):
            SessionManager.create_session(config)

    @patch("snowflake_ml_template.core.session.manager.Session")
    def test_get_session_returns_current_session(
        self, mock_session_class: Mock
    ) -> None:
        """Test that get_session returns the current session."""
        # Setup mock
        mock_session = Mock()
        mock_builder = Mock()
        mock_builder.configs.return_value = mock_builder
        mock_builder.create.return_value = mock_session
        mock_session_class.builder = mock_builder

        # Create session
        config = {
            "account": "test_account",
            "user": "test_user",
            "password": "test_password",
        }
        SessionManager.create_session(config)

        # Get session
        session = SessionManager.get_session()

        assert session == mock_session

    def test_get_session_without_create_raises_error(self) -> None:
        """Test that get_session raises error if no session exists."""
        with pytest.raises(RuntimeError, match="No active session"):
            SessionManager.get_session()

    @patch("snowflake_ml_template.core.session.manager.Session")
    def test_switch_warehouse(self, mock_session_class: Mock) -> None:
        """Test warehouse switching."""
        # Setup mock
        mock_session = Mock()
        mock_builder = Mock()
        mock_builder.configs.return_value = mock_builder
        mock_builder.create.return_value = mock_session
        mock_session_class.builder = mock_builder

        # Create session
        config = {"account": "test_account", "user": "test_user", "warehouse": "WH1"}
        SessionManager.create_session(config)

        # Switch warehouse
        SessionManager.switch_warehouse("WH2")

        mock_session.use_warehouse.assert_called_once_with("WH2")

    def test_switch_warehouse_without_session_raises_error(self) -> None:
        """Test that switch_warehouse raises error if no session exists."""
        with pytest.raises(RuntimeError, match="No active session"):
            SessionManager.switch_warehouse("WH2")

    def test_switch_warehouse_empty_name_raises_error(self) -> None:
        """Test that empty warehouse name raises ValueError."""
        # Create a mock session first
        with patch("snowflake_ml_template.core.session.manager.Session"):
            SessionManager._instance = SessionManager()
            SessionManager._instance._current_session = Mock()

            with pytest.raises(ValueError, match="Warehouse name cannot be empty"):
                SessionManager.switch_warehouse("")

    @patch("snowflake_ml_template.core.session.manager.Session")
    def test_switch_database(self, mock_session_class: Mock) -> None:
        """Test database switching."""
        # Setup mock
        mock_session = Mock()
        mock_builder = Mock()
        mock_builder.configs.return_value = mock_builder
        mock_builder.create.return_value = mock_session
        mock_session_class.builder = mock_builder

        # Create session
        config = {"account": "test_account", "user": "test_user", "database": "DB1"}
        SessionManager.create_session(config)

        # Switch database
        SessionManager.switch_database("DB2", "SCHEMA2")

        mock_session.use_database.assert_called_once_with("DB2")
        mock_session.use_schema.assert_called_once_with("SCHEMA2")

    @patch("snowflake_ml_template.core.session.manager.Session")
    def test_close_session(self, mock_session_class: Mock) -> None:
        """Test session closing."""
        # Setup mock
        mock_session = Mock()
        mock_builder = Mock()
        mock_builder.configs.return_value = mock_builder
        mock_builder.create.return_value = mock_session
        mock_session_class.builder = mock_builder

        # Create session
        config = {"account": "test_account", "user": "test_user"}
        SessionManager.create_session(config)

        # Close session
        SessionManager.close_session()

        mock_session.close.assert_called_once()


class TestSessionContext:
    """Test cases for SessionContext class."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        SessionManager._instance = None

    def test_initialization_with_warehouse(self) -> None:
        """Test SessionContext initialization with warehouse."""
        context = SessionContext(warehouse="TEST_WH")

        assert context.warehouse == "TEST_WH"
        assert context.database is None
        assert context.schema is None

    def test_initialization_with_database_and_schema(self) -> None:
        """Test SessionContext initialization with database and schema."""
        context = SessionContext(database="TEST_DB", schema="TEST_SCHEMA")

        assert context.warehouse is None
        assert context.database == "TEST_DB"
        assert context.schema == "TEST_SCHEMA"

    def test_initialization_schema_without_database_raises_error(self) -> None:
        """Test that schema without database raises ValueError."""
        with pytest.raises(
            ValueError, match="Schema cannot be specified without database"
        ):
            SessionContext(schema="TEST_SCHEMA")

    @patch("snowflake_ml_template.core.session.manager.Session")
    def test_context_manager_switches_warehouse(self, mock_session_class: Mock) -> None:
        """Test that context manager switches warehouse."""
        # Setup mock
        mock_session = Mock()
        mock_builder = Mock()
        mock_builder.configs.return_value = mock_builder
        mock_builder.create.return_value = mock_session
        mock_session_class.builder = mock_builder

        # Create session
        config = {"account": "test_account", "user": "test_user", "warehouse": "WH1"}
        SessionManager.create_session(config)

        # Mock get_current_warehouse
        with patch.object(SessionManager, "get_current_warehouse", return_value="WH1"):
            # Use context manager
            with SessionContext(warehouse="WH2"):
                # Verify warehouse was switched
                assert mock_session.use_warehouse.called
                mock_session.use_warehouse.assert_called_with("WH2")

            # Verify warehouse was restored
            calls = mock_session.use_warehouse.call_args_list
            assert len(calls) == 2
            assert calls[1][0][0] == "WH1"  # Restored to WH1

    def test_repr(self) -> None:
        """Test string representation."""
        context = SessionContext(warehouse="TEST_WH", database="TEST_DB")

        repr_str = repr(context)

        assert "warehouse=TEST_WH" in repr_str
        assert "database=TEST_DB" in repr_str
