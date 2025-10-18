"""Unit tests for feature store core."""

import pytest

from snowflake_ml_template.core.exceptions.errors import FeatureStoreError
from snowflake_ml_template.feature_store.core.entity import Entity
from snowflake_ml_template.feature_store.core.store import FeatureStore


class StubSQL:
    """Stub SQL."""

    def __init__(self, behavior):
        """Stub init."""
        # behavior: callable(query) -> list or raises
        self._behavior = behavior

    def collect(self):
        """Stub collect."""
        result = self._behavior()
        return result


class StubSession:
    """Stub session."""

    def __init__(self, plan):
        """Stub init."""
        # plan: list of tuples (predicate_substring, return_list, raise_exc)
        self.plan = plan
        self.queries = []

    def sql(self, query: str):
        """Stub sql."""
        self.queries.append(query)

        def behave():
            """Stub behave."""
            for substr, rows, raise_exc in self.plan:
                if substr in query:
                    if raise_exc:
                        raise RuntimeError("boom")
                    return rows
            return []

        return StubSQL(behave)


class DummyFeatureDFWriter:
    """Define dummy feature DataFrame writer."""

    def __init__(self, rec):
        """Initialize writer."""
        self.rec = rec

    def mode(self, m):
        """Set write mode."""
        self.rec.append(("mode", m))
        return self

    def save_as_table(self, full_name):
        """Save as table."""
        self.rec.append(("save_as_table", full_name))


class DummyFeatureDF:
    """Define dummy feature DataFrame."""

    def __init__(self, rec):
        """Initialize DataFrame."""
        self.write = DummyFeatureDFWriter(rec)


class DummyFeatureView:
    """Define dummy feature view."""

    def __init__(self, name, version, managed, rec):
        """Initialize feature view."""
        self.name = name
        self.version = version
        self.is_snowflake_managed = managed
        self.feature_df = DummyFeatureDF(rec)


def test_init_validation_errors():
    """Test init validation errors."""
    with pytest.raises(ValueError):
        FeatureStore(session=None, database="DB")
    with pytest.raises(ValueError):
        FeatureStore(session=StubSession([]), database="")


def test_ensure_schema_exists_error_raises_feature_store_error():
    """Test ensure schema exists error raises feature store error."""
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], True),
        ]
    )
    with pytest.raises(FeatureStoreError):
        FeatureStore(session=sess, database="DB", schema="FEAT")


def test_register_entity_creates_metadata_and_updates():
    """Test register entity creates metadata and updates."""
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("CREATE TABLE IF NOT EXISTS", [], False),
        ]
    )
    fs = FeatureStore(session=sess, database="DB", schema="FEAT")
    e = Entity(name="CUSTOMER", join_keys=["CUST_ID"])
    fs.register_entity(e)
    # verify table creation query issued
    assert any(
        "CREATE TABLE IF NOT EXISTS DB.FEAT.ENTITY_CUSTOMER_METADATA" in q
        for q in sess.queries
    )
    # register again to trigger warning path and update
    fs.register_entity(e)
    assert fs.get_entity("CUSTOMER").name == "CUSTOMER"
    assert "CUSTOMER" in fs.list_entities()


def test_register_feature_view_managed_creates_table_and_overwrite():
    """Test register feature view managed creates table and overwrite."""
    rec = []
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("DROP TABLE IF EXISTS", [], False),
            ("CREATE TABLE IF NOT EXISTS", [], False),
        ]
    )
    fs = FeatureStore(session=sess, database="DB", schema="FEAT")
    fv = DummyFeatureView("FV", "1.2.3", True, rec)

    # fresh register
    fs.register_feature_view(fv, overwrite=False)
    # managed path writes table
    assert ("mode", "overwrite") in rec
    assert any(
        name == "save_as_table" and full.endswith("DB.FEAT.FEATURE_VIEW_FV_V1_2_3")
        for (name, full) in rec
    )
    # metadata table created
    assert any(
        "CREATE TABLE IF NOT EXISTS DB.FEAT.FEATURE_VIEW_FV_METADATA" in q
        for q in sess.queries
    )

    # overwrite path
    fs.register_feature_view(fv, overwrite=True)
    assert any(
        "DROP TABLE IF EXISTS DB.FEAT.FEATURE_VIEW_FV_V1_2_3" in q for q in sess.queries
    )


def test_register_feature_view_conflict_without_overwrite_raises():
    """Test register feature view conflict without overwrite raises."""
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("CREATE TABLE IF NOT EXISTS", [], False),
        ]
    )
    fs = FeatureStore(session=sess, database="DB", schema="FEAT")
    rec = []
    fv = DummyFeatureView("FV", "1.0.0", True, rec)
    fs.register_feature_view(fv)
    with pytest.raises(FeatureStoreError):
        fs.register_feature_view(fv, overwrite=False)


def test_register_feature_view_external_validation_paths():
    """Test register feature view external validation paths."""
    # success path: SHOW TABLES returns non-empty
    sess1 = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            (
                "SHOW TABLES LIKE 'FEATURE_VIEW_FV_V1_0_0' IN SCHEMA DB.FEAT",
                [{"name": "ok"}],
                False,
            ),
            ("CREATE TABLE IF NOT EXISTS", [], False),
        ]
    )
    fs1 = FeatureStore(session=sess1, database="DB", schema="FEAT")
    rec1 = []
    fv1 = DummyFeatureView("FV", "1.0.0", False, rec1)
    fs1.register_feature_view(fv1)

    # missing table path
    sess2 = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("SHOW TABLES LIKE 'FEATURE_VIEW_FV_V2_0_0' IN SCHEMA DB.FEAT", [], False),
        ]
    )
    fs2 = FeatureStore(session=sess2, database="DB", schema="FEAT")
    fv2 = DummyFeatureView("FV", "2.0.0", False, [])
    with pytest.raises(FeatureStoreError):
        fs2.register_feature_view(fv2)

    # exception path during SHOW TABLES
    sess3 = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("SHOW TABLES LIKE 'FEATURE_VIEW_FV_V3_0_0' IN SCHEMA DB.FEAT", [], True),
        ]
    )
    fs3 = FeatureStore(session=sess3, database="DB", schema="FEAT")
    fv3 = DummyFeatureView("FV", "3.0.0", False, [])
    with pytest.raises(FeatureStoreError):
        fs3.register_feature_view(fv3)

    # getters
    assert fs1.get_feature_view("FV").name == "FV"
    assert "FV" in fs1.list_feature_views()
