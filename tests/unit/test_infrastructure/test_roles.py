"""Unit tests for roles."""

import pytest

from snowflake_ml_template.core.exceptions.errors import ConfigurationError
from snowflake_ml_template.infrastructure.provisioning.roles import RoleProvisioner


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


def test_create_mlops_roles_calls_create_role():
    """Test create mlops roles calls create role."""
    sess = StubSession(
        [
            ("CREATE ROLE IF NOT EXISTS", lambda: [], False),
        ]
    )
    rp = RoleProvisioner(sess)
    res = rp.create_mlops_roles()
    assert all(res.values())


def test_create_role_success_and_failure():
    """Test create role success and failure."""
    sess_ok = StubSession([("CREATE ROLE IF NOT EXISTS", lambda: [], False)])
    rp_ok = RoleProvisioner(sess_ok)
    assert rp_ok.create_role("R1", comment="c") is True

    sess_fail = StubSession([('CREATE ROLE IF NOT EXISTS "BAD"', None, True)])
    rp_fail = RoleProvisioner(sess_fail)
    with pytest.raises(ConfigurationError):
        rp_fail.create_role("BAD")


def test_grants_success_and_failure():
    """Test grants success and failure."""
    # grant to user
    sess_user_ok = StubSession([("GRANT ROLE", lambda: [], False)])
    rp_user_ok = RoleProvisioner(sess_user_ok)
    assert rp_user_ok.grant_role_to_user("R", "U") is True

    sess_user_fail = StubSession([("GRANT ROLE", None, True)])
    rp_user_fail = RoleProvisioner(sess_user_fail)
    with pytest.raises(ConfigurationError):
        rp_user_fail.grant_role_to_user("R", "U")

    # grant role to role
    sess_role_ok = StubSession([("GRANT ROLE", lambda: [], False)])
    rp_role_ok = RoleProvisioner(sess_role_ok)
    assert rp_role_ok.grant_role_to_role("C", "P") is True

    sess_role_fail = StubSession([("GRANT ROLE", None, True)])
    rp_role_fail = RoleProvisioner(sess_role_fail)
    with pytest.raises(ConfigurationError):
        rp_role_fail.grant_role_to_role("C", "P")

    # database privileges
    sess_db_ok = StubSession([("GRANT USAGE ON DATABASE", lambda: [], False)])
    rp_db_ok = RoleProvisioner(sess_db_ok)
    assert rp_db_ok.grant_database_privileges("R", "DB", ["USAGE"]) is True

    sess_db_fail = StubSession([("GRANT USAGE ON DATABASE", None, True)])
    rp_db_fail = RoleProvisioner(sess_db_fail)
    with pytest.raises(ConfigurationError):
        rp_db_fail.grant_database_privileges("R", "DB", ["USAGE"])

    # schema privileges
    sess_sc_ok = StubSession([("GRANT USAGE ON SCHEMA", lambda: [], False)])
    rp_sc_ok = RoleProvisioner(sess_sc_ok)
    assert rp_sc_ok.grant_schema_privileges("R", "DB", "SC", ["USAGE"]) is True

    sess_sc_fail = StubSession([("GRANT USAGE ON SCHEMA", None, True)])
    rp_sc_fail = RoleProvisioner(sess_sc_fail)
    with pytest.raises(ConfigurationError):
        rp_sc_fail.grant_schema_privileges("R", "DB", "SC", ["USAGE"])

    # warehouse privileges
    sess_wh_ok = StubSession([("GRANT USAGE ON WAREHOUSE", lambda: [], False)])
    rp_wh_ok = RoleProvisioner(sess_wh_ok)
    assert rp_wh_ok.grant_warehouse_privileges("R", "WH", ["USAGE"]) is True

    sess_wh_fail = StubSession([("GRANT USAGE ON WAREHOUSE", None, True)])
    rp_wh_fail = RoleProvisioner(sess_wh_fail)
    with pytest.raises(ConfigurationError):
        rp_wh_fail.grant_warehouse_privileges("R", "WH", ["USAGE"])


def test_role_exists_and_list_roles():
    """Test role exists and list roles."""
    sess = StubSession(
        [
            ("SHOW ROLES LIKE", lambda: [{"name": "R"}], False),
            ("SHOW ROLES", lambda: [{"name": "A"}, {"name": "B"}], False),
        ]
    )
    rp = RoleProvisioner(sess)
    assert rp.role_exists("R") is True
    assert rp.list_roles() == ["A", "B"]
