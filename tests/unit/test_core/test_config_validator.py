"""Unit tests for config validator."""

import pytest

from snowflake_ml_template.core.config.models import (
    FeatureStoreConfig,
    ModelRegistryConfig,
    SnowflakeConfig,
)
from snowflake_ml_template.core.config.validator import ConfigValidator


class StubSQL:
    """Stub SQL."""

    def __init__(self, rows, raise_exc=False):
        """Stub init."""
        self._rows = rows
        self._raise = raise_exc

    def collect(self):
        """Stub collect."""
        if self._raise:
            raise RuntimeError("boom")
        return self._rows


class StubSession:
    """Stub session."""

    def __init__(self, responses):
        """Stub init."""
        # responses: list of tuples (predicate_substring, rows, raise_exc)
        self._responses = responses

    def sql(self, query: str):
        """Stub sql."""
        for substr, rows, raise_exc in self._responses:
            if substr in query:
                return StubSQL(rows, raise_exc)
        # default: empty
        return StubSQL([])


@pytest.fixture
def validator_success():
    """Stub validator success."""
    # default success: SHOW ... returns non-empty list
    sess = StubSession(
        [
            ("SHOW DATABASES LIKE", [{"name": "DB"}], False),
            ("SHOW SCHEMAS LIKE", [{"name": "SC"}], False),
            ("SHOW WAREHOUSES LIKE", [{"name": "WH"}], False),
            ("SHOW TABLES LIKE", [{"name": "TBL"}], False),
            ("SHOW STAGES LIKE", [{"name": "STG"}], False),
            ("SHOW DATABASES", [{"name": "DB1"}, {"name": "DB2"}], False),
            ("SHOW SCHEMAS IN DATABASE", [{"name": "PUBLIC"}], False),
            ("SHOW WAREHOUSES", [{"name": "WH"}], False),
            ("SHOW TABLES IN SCHEMA", [{"name": "FOO"}], False),
        ]
    )
    return ConfigValidator(sess)


def test_validate_database_success(validator_success):
    """Test validate database success."""
    assert validator_success.validate_database("DB") is True


def test_validate_database_not_found():
    """Test validate database not found."""
    sess = StubSession([("SHOW DATABASES LIKE", [], False)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_database("MISSING")


def test_validate_database_exception_wrap():
    """Test validate database exception wrap."""
    sess = StubSession([("SHOW DATABASES LIKE", [], True)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_database("ERR")


def test_validate_schema_success(validator_success):
    """Test validate schema success."""
    assert validator_success.validate_schema("DB", "SC") is True


def test_validate_schema_not_found():
    """Test validate schema not found."""
    sess = StubSession([("SHOW SCHEMAS LIKE", [], False)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_schema("DB", "SC")


def test_validate_schema_exception_wrap():
    """Test validate schema exception wrap."""
    sess = StubSession([("SHOW SCHEMAS LIKE", [], True)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_schema("DB", "SC")


def test_validate_warehouse_success(validator_success):
    """Test validate warehouse success."""
    assert validator_success.validate_warehouse("WH") is True


def test_validate_warehouse_not_found():
    """Test validate warehouse not found."""
    sess = StubSession([("SHOW WAREHOUSES LIKE", [], False)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_warehouse("MISSING")


def test_validate_warehouse_exception_wrap():
    """Test validate warehouse exception wrap."""
    sess = StubSession([("SHOW WAREHOUSES LIKE", [], True)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_warehouse("WH")


def test_validate_table_success(validator_success):
    """Test validate table success."""
    assert validator_success.validate_table("DB", "SC", "T") is True


def test_validate_table_not_found():
    """Test validate table not found."""
    sess = StubSession([("SHOW TABLES LIKE", [], False)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_table("DB", "SC", "T")


def test_validate_table_exception_wrap():
    """Test validate table exception wrap."""
    sess = StubSession([("SHOW TABLES LIKE", [], True)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_table("DB", "SC", "T")


def test_validate_stage_success(validator_success):
    """Test validate stage success."""
    assert validator_success.validate_stage("@STG") is True


def test_validate_stage_not_found():
    """Test validate stage not found."""
    sess = StubSession([("SHOW STAGES LIKE", [], False)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_stage("@MISSING")


def test_validate_stage_exception_wrap():
    """Test validate stage exception wrap."""
    sess = StubSession([("SHOW STAGES LIKE", [], True)])
    v = ConfigValidator(sess)
    with pytest.raises(ValueError):
        v.validate_stage("@ERR")


def test_get_lists(validator_success):
    """Test get lists."""
    assert validator_success.get_database_list() == ["DB1", "DB2"]
    assert validator_success.get_schema_list("DB") == ["PUBLIC"]
    assert validator_success.get_warehouse_list() == ["WH"]
    assert validator_success.get_table_list("DB", "SC") == ["FOO"]


def test_validate_snowflake_config_calls_resource_checks(monkeypatch):
    """Test validate snowflake config calls resource checks."""
    # Prepare validator with dummy session (won't be used due to monkeypatches)
    v = ConfigValidator(StubSession([]))

    called = {"wh": False, "db": False, "sc": False}

    def vw(x):
        called["wh"] = True
        return True

    def vd(x):
        called["db"] = True
        return True

    def vs(db, sc):
        called["sc"] = True
        return True

    monkeypatch.setattr(ConfigValidator, "validate_warehouse", staticmethod(vw))
    monkeypatch.setattr(ConfigValidator, "validate_database", staticmethod(vd))
    monkeypatch.setattr(ConfigValidator, "validate_schema", staticmethod(vs))

    cfg = SnowflakeConfig(
        account="acc",
        user="usr",
        warehouse="WH",
        database="DB",
        schema="SC",  # alias to schema_
    )
    assert v.validate_snowflake_config(cfg, check_resources=True) is True
    assert all(called.values())


def test_feature_store_and_registry_validators_use_schema(monkeypatch):
    """Test feature store and registry validators use schema."""
    v = ConfigValidator(
        StubSession(
            [
                ("SHOW DATABASES LIKE", [{"name": "DB"}], False),
                ("SHOW SCHEMAS LIKE", [{"name": "SC"}], False),
            ]
        )
    )
    fs = FeatureStoreConfig(database="DB", schema="SC")
    mr = ModelRegistryConfig(database="DB", schema="SC")
    assert v.validate_feature_store_config(fs) is True
    assert v.validate_model_registry_config(mr) is True
