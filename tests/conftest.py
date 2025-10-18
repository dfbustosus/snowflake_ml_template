"""Pytest configuration and fixtures."""

from unittest.mock import Mock

import pytest
from snowflake.snowpark import Session


@pytest.fixture
def mock_session():
    """Create a mock Snowflake session."""
    session = Mock(spec=Session)
    session.sql = Mock(return_value=Mock(collect=Mock(return_value=[])))
    session.table = Mock(return_value=Mock())
    return session


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "account": "test_account",
        "user": "test_user",
        "password": "test_password",
        "warehouse": "TEST_WH",
        "database": "TEST_DB",
        "schema": "TEST_SCHEMA",
    }
