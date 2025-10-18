"""Unit tests for database provisioning."""

import pytest

from snowflake_ml_template.core.exceptions.errors import ConfigurationError
from snowflake_ml_template.infrastructure.provisioning.databases import (
    DatabaseProvisioner,
)


class StubSQL:
    """Stub SQL."""

    def __init__(self, behavior):
        """Stub SQL."""
        self._behavior = behavior
        self._binds = []

    def bind(self, *args):  # pragma: no cover (not used here)
        """Stub bind."""
        self._binds.extend(args)
        return self

    def collect(self):
        """Stub collect."""
        return self._behavior()


class StubSession:
    """Stub session."""

    def __init__(self, routes):
        """Stub session."""
        # routes: list of (substring, callable returning list, raise_exc)
        self.routes = routes
        self.queries = []

    def sql(self, query: str):
        """Stub SQL query."""
        self.queries.append(query)

        def behavior():
            """Stub behavior."""
            for substr, fn, raise_exc in self.routes:
                if substr in query:
                    if raise_exc:
                        raise RuntimeError("boom")
                    return fn() if fn else []
            return []

        return StubSQL(behavior)


def test_create_environment_databases_happy_and_clone():
    """Test create environment databases happy and clone."""
    sess = StubSession(
        [
            ("CREATE DATABASE IF NOT EXISTS", lambda: [], False),
            ("CREATE OR REPLACE DATABASE", lambda: [], False),
        ]
    )
    prov = DatabaseProvisioner(sess)
    res = prov.create_environment_databases(clone_test_from_prod=True)
    assert res["ML_DEV_DB"] and res["ML_TEST_DB"] and res["ML_PROD_DB"]


def test_create_database_variants_and_failure():
    """Test create database variants and failure."""
    # success with transient and comment
    sess_ok = StubSession(
        [
            ("CREATE TRANSIENT DATABASE IF NOT EXISTS", lambda: [], False),
        ]
    )
    prov_ok = DatabaseProvisioner(sess_ok)
    assert (
        prov_ok.create_database(
            "DB", comment="c", transient=True, data_retention_time_in_days=7
        )
        is True
    )

    # failure raises ConfigurationError
    sess_fail = StubSession(
        [
            ('CREATE DATABASE IF NOT EXISTS "BAD"', lambda: [], True),
        ]
    )
    prov_fail = DatabaseProvisioner(sess_fail)
    with pytest.raises(ConfigurationError):
        prov_fail.create_database("BAD")


def test_clone_and_drop_and_exists_and_list():
    """Test clone and drop and exists and list."""
    sess = StubSession(
        [
            ("CREATE OR REPLACE DATABASE", lambda: [], False),
            ("DROP DATABASE", lambda: [], False),
            ("SHOW DATABASES LIKE", lambda: [{"name": "DB"}], False),
            ("SHOW DATABASES", lambda: [{"name": "A"}, {"name": "B"}], False),
        ]
    )
    prov = DatabaseProvisioner(sess)
    assert prov.clone_database("SRC", "TGT", comment="copy") is True
    assert prov.drop_database("TGT", if_exists=True) is True
    assert prov.database_exists("DB") is True
    assert prov.list_databases() == ["A", "B"]


def test_database_exists_exception_and_drop_error():
    """Test database exists exception and drop error."""
    sess = StubSession(
        [
            ("SHOW DATABASES LIKE", None, True),
            ("DROP DATABASE", None, True),
        ]
    )
    prov = DatabaseProvisioner(sess)
    assert prov.database_exists("X") is False
    with pytest.raises(ConfigurationError):
        prov.drop_database("X")


def test_create_database_with_tags_and_replication():
    """Ensure tagging and replication statements are executed."""
    sess = StubSession(
        [
            ("CREATE DATABASE IF NOT EXISTS", lambda: [], False),
            ("ALTER DATABASE", lambda: [], False),
            ("ENABLE REPLICATION", lambda: [], False),
        ]
    )
    prov = DatabaseProvisioner(sess)
    prov.create_database(
        name="ANALYTICS_DB",
        tags={"ml.tag": "analytics"},
        replication_targets=["ORG.ACCOUNT1", "ORG.ACCOUNT2"],
    )
    assert any('ALTER DATABASE "ANALYTICS_DB" SET TAG' in q for q in sess.queries)
    assert any("ENABLE REPLICATION" in q for q in sess.queries)
