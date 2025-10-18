"""Unit tests for warehouses."""

import pytest

from snowflake_ml_template.core.exceptions import ConfigurationError
from snowflake_ml_template.infrastructure.provisioning.warehouses import (
    WarehouseProvisioner,
)


class SQLStub:
    """Stub SQL."""

    def __init__(self, behavior):
        """Stub SQL."""
        self._behavior = behavior

    def collect(self):
        """Stub collect."""
        return self._behavior()


class SessionStub:
    """Stub session."""

    def __init__(self, routes):
        """Stub session."""
        # routes: list of (substring, callable returning list)
        self.routes = routes
        self.queries = []

    def sql(self, query: str):
        """Stub SQL query."""
        self.queries.append(query)
        for substr, beh in self.routes:
            if substr in query:
                return SQLStub(beh)
        return SQLStub(lambda: [])


def test_create_mlops_warehouses_calls_create_warehouse_for_all():
    """Test create mlops warehouses calls create warehouse for all."""
    created = []

    # Intercept CREATE WAREHOUSE calls
    def create_beh():
        created.append(1)
        return []

    sess = SessionStub([("CREATE WAREHOUSE IF NOT EXISTS", create_beh)])
    prov = WarehouseProvisioner(sess)
    res = prov.create_mlops_warehouses()
    assert all(res.values())
    # At least four warehouses attempted
    assert len(created) >= 4


def test_create_warehouse_success_and_sql_composition():
    """Test create warehouse success and sql composition."""
    sess = SessionStub([("CREATE WAREHOUSE IF NOT EXISTS", lambda: [])])
    prov = WarehouseProvisioner(sess)
    ok = prov.create_warehouse(
        name="TRAIN_WH",
        size="LARGE",
        type="SNOWPARK-OPTIMIZED",
        auto_suspend=30,
        auto_resume=False,
        min_cluster_count=2,
        max_cluster_count=4,
        comment="training",
        statement_timeout_in_seconds=120,
        tags={"ml.stage": "train"},
        resource_monitor="RM",
        scaling_policy="ECONOMY",
        grants_to_roles=["ML_ENGINEER"],
    )
    assert ok is True
    q = " ".join(sess.queries)
    assert "WAREHOUSE_SIZE = 'LARGE'" in q
    assert "WAREHOUSE_TYPE = 'SNOWPARK-OPTIMIZED'" in q
    assert "AUTO_SUSPEND = 30" in q
    assert "AUTO_RESUME = FALSE" in q
    assert "MIN_CLUSTER_COUNT = 2" in q
    assert "MAX_CLUSTER_COUNT = 4" in q
    assert "COMMENT = 'training'" in q
    assert "STATEMENT_TIMEOUT_IN_SECONDS = 120" in q
    assert "SCALING_POLICY = 'ECONOMY'" in q
    assert 'RESOURCE_MONITOR = "RM"' in q
    assert any('SET TAG "ml"."stage" = ' in query for query in sess.queries)
    assert any(
        'GRANT OWNERSHIP ON WAREHOUSE "TRAIN_WH" TO ROLE "ML_ENGINEER"' in query
        for query in sess.queries
    )


def test_create_warehouse_failure_raises_configuration_error():
    """Test create warehouse failure raises configuration error."""
    sess = SessionStub(
        [
            (
                "CREATE WAREHOUSE IF NOT EXISTS",
                lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            )
        ]
    )
    prov = WarehouseProvisioner(sess)
    with pytest.raises(ConfigurationError):
        prov.create_warehouse("BAD_WH")


def test_warehouse_exists_true_false_exception():
    """Test warehouse exists true false exception."""
    # True
    sess_true = SessionStub([("SHOW WAREHOUSES LIKE", lambda: [{"name": "X"}])])
    prov_true = WarehouseProvisioner(sess_true)
    assert prov_true.warehouse_exists("X") is True
    # False
    sess_false = SessionStub([("SHOW WAREHOUSES LIKE", lambda: [])])
    prov_false = WarehouseProvisioner(sess_false)
    assert prov_false.warehouse_exists("X") is False
    # Exception -> False
    sess_err = SessionStub(
        [("SHOW WAREHOUSES LIKE", lambda: (_ for _ in ()).throw(RuntimeError("err")))]
    )
    prov_err = WarehouseProvisioner(sess_err)
    assert prov_err.warehouse_exists("X") is False


def test_list_warehouses_success_and_exception_path():
    """Test list warehouses success and exception path."""
    sess = SessionStub([("SHOW WAREHOUSES", lambda: [{"name": "A"}, {"name": "B"}])])
    prov = WarehouseProvisioner(sess)
    assert prov.list_warehouses() == ["A", "B"]

    sess2 = SessionStub(
        [("SHOW WAREHOUSES", lambda: (_ for _ in ()).throw(RuntimeError("fail")))]
    )
    prov2 = WarehouseProvisioner(sess2)
    assert prov2.list_warehouses() == []
