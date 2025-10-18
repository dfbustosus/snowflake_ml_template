"""Unit tests for schemas."""

import pytest

from snowflake_ml_template.core.exceptions.errors import ConfigurationError
from snowflake_ml_template.infrastructure.provisioning.schemas import SchemaProvisioner


class StubSQL:
    """Stub SQL."""

    def __init__(self, behavior):
        """Stub SQL."""
        self._behavior = behavior

    def collect(self):
        """Stub collect."""
        return self._behavior()


class StubSession:
    """Stub session."""

    def __init__(self, routes):
        """Stub session."""
        # routes: list of (substring, fn, raise_exc)
        self.routes = routes
        self.queries = []

    def sql(self, query: str):
        """Stub SQL query."""
        self.queries.append(query)

        def behavior():
            for substr, fn, raise_exc in self.routes:
                if substr in query:
                    if raise_exc:
                        raise RuntimeError("boom")
                    return fn() if fn else []
            return []

        return StubSQL(behavior)


def test_create_canonical_schemas_and_exists_and_list():
    """Test create canonical schemas and exists and list."""
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", lambda: [], False),
            ("SHOW SCHEMAS LIKE", lambda: [{"name": "FEATURES"}], False),
            (
                "SHOW SCHEMAS IN DATABASE",
                lambda: [{"name": "RAW_DATA"}, {"name": "FEATURES"}],
                False,
            ),
        ]
    )
    sp = SchemaProvisioner(sess)
    res = sp.create_canonical_schemas("DB")
    assert all(res.values())
    assert sp.schema_exists("DB", "FEATURES") is True
    assert sp.list_schemas("DB") == ["RAW_DATA", "FEATURES"]


def test_create_schema_with_flags_and_failure():
    """Test create schema with flags and failure."""
    # success with transient and managed access and comment
    sess_ok = StubSession(
        [
            ("CREATE TRANSIENT SCHEMA IF NOT EXISTS", lambda: [], False),
        ]
    )
    sp_ok = SchemaProvisioner(sess_ok)
    assert (
        sp_ok.create_schema(
            "DB", "EXPERIMENTS", comment="c", transient=True, managed_access=True
        )
        is True
    )

    # failure raises ConfigurationError
    sess_fail = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS DB.BAD", None, True),
        ]
    )
    sp_fail = SchemaProvisioner(sess_fail)
    with pytest.raises(ConfigurationError):
        sp_fail.create_schema("DB", "BAD")


def test_drop_schema_cascade_success_and_error():
    """Test drop schema cascade success and error."""
    sess_ok = StubSession(
        [
            ("DROP SCHEMA IF EXISTS", lambda: [], False),
        ]
    )
    sp_ok = SchemaProvisioner(sess_ok)
    assert sp_ok.drop_schema("DB", "SC", if_exists=True, cascade=True) is True

    sess_fail = StubSession(
        [
            ("DROP SCHEMA IF EXISTS", None, True),
        ]
    )
    sp_fail = SchemaProvisioner(sess_fail)
    with pytest.raises(ConfigurationError):
        sp_fail.drop_schema("DB", "SC", if_exists=True, cascade=False)


def test_schema_exists_exception_and_list_exception():
    """Test schema exists exception and list exception."""
    sess = StubSession(
        [
            ("SHOW SCHEMAS LIKE", None, True),
            ("SHOW SCHEMAS IN DATABASE", None, True),
        ]
    )
    sp = SchemaProvisioner(sess)
    assert sp.schema_exists("DB", "X") is False
    assert sp.list_schemas("DB") == []
