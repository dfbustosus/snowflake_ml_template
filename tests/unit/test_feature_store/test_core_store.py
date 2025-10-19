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


class StubFeatureStoreClient:
    """Stub feature store client capturing invocations."""

    def __init__(self):
        """Stub init."""
        self.entity_calls = []
        self.fv_calls = []
        self.refresh_history = []

    def create_or_replace_entity(self, **kwargs):  # pragma: no cover - simple record
        """Stub create or replace entity."""
        self.entity_calls.append(("create_or_replace_entity", kwargs))

    def create_or_replace_feature_view(self, **kwargs):  # pragma: no cover - record
        """Stub create or replace feature view."""
        self.fv_calls.append(("create_or_replace_feature_view", kwargs))


class StubSession:
    """Stub session."""

    def __init__(self, plan):
        """Stub init."""
        # plan: list of tuples (predicate_substring, return_list, raise_exc)
        self.plan = plan
        self.queries = []
        self.feature_store = StubFeatureStoreClient()

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


class DummyFeatureDF:
    """Define dummy feature DataFrame providing SQL text."""

    def __init__(self, sql_text):
        """Stub init."""
        self._sql_text = sql_text

    def to_sql(self):
        """Stub to_sql."""
        return self._sql_text


class DummyFeatureView:
    """Define dummy feature view."""

    def __init__(self, name, version, managed, feature_sql):
        """Initialize feature view."""
        self.name = name
        self.version = version
        self.is_snowflake_managed = managed
        self.feature_df = DummyFeatureDF(feature_sql)
        self.entities = []
        self.feature_names = ["f1", "f2"]
        self.refresh_freq = "1 hour" if managed else None
        self.description = ""
        self.tags = None

    @property
    def is_external(self):
        """Stub is_external."""
        return not self.is_snowflake_managed


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
        ]
    )
    fs = FeatureStore(session=sess, database="DB", schema="FEAT")
    e = Entity(name="CUSTOMER", join_keys=["CUST_ID"])
    fs.register_entity(e)
    # verify client invocation
    assert sess.feature_store.entity_calls[0][0] == "create_or_replace_entity"
    assert sess.feature_store.entity_calls[0][1]["name"] == "CUSTOMER"
    # register again (should reuse cached entity)
    fs.register_entity(e)
    assert fs.get_entity("CUSTOMER").name == "CUSTOMER"
    assert "CUSTOMER" in fs.list_entities()


def test_register_feature_view_managed_creates_table_and_overwrite():
    """Test register feature view managed creates table and overwrite."""
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("DROP DYNAMIC TABLE IF EXISTS", [], False),
            ("CREATE OR REPLACE DYNAMIC TABLE", [], False),
        ]
    )
    fs = FeatureStore(session=sess, database="DB", schema="FEAT")
    fv = DummyFeatureView("FV", "1.2.3", True, "SELECT 1 AS f1, 2 AS f2 FROM DUAL")
    fv.entities = [Entity(name="CUSTOMER", join_keys=["ID"])]

    # fresh register
    fs.register_feature_view(fv, overwrite=False)
    assert any("CREATE OR REPLACE DYNAMIC TABLE" in q for q in sess.queries)
    assert sess.feature_store.fv_calls[0][1]["name"] == "FV"

    # overwrite path
    fs.register_feature_view(fv, overwrite=True)
    assert any(
        "DROP DYNAMIC TABLE IF EXISTS DB.FEAT.FEATURE_VIEW_FV_V1_2_3" in q
        for q in sess.queries
    )


def test_register_feature_view_conflict_without_overwrite_raises():
    """Test register feature view conflict without overwrite raises."""
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("CREATE OR REPLACE DYNAMIC TABLE", [], False),
        ]
    )
    fs = FeatureStore(session=sess, database="DB", schema="FEAT")
    fv = DummyFeatureView("FV", "1.0.0", True, "SELECT 1")
    fv.entities = [Entity(name="CUSTOMER", join_keys=["ID"])]
    fs.register_feature_view(fv)
    sess.feature_store.fv_calls.clear()
    with pytest.raises(FeatureStoreError):
        fs.register_feature_view(fv, overwrite=False)


def test_dynamic_table_lifecycle_helpers():
    """Ensure suspend/resume/refresh execute expected SQL."""
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("CREATE OR REPLACE DYNAMIC TABLE", [], False),
            ("ALTER DYNAMIC TABLE DB.FEAT.FEATURE_VIEW_FV_V1_0_0 SUSPEND", [], False),
            ("ALTER DYNAMIC TABLE DB.FEAT.FEATURE_VIEW_FV_V1_0_0 RESUME", [], False),
            ("ALTER DYNAMIC TABLE DB.FEAT.FEATURE_VIEW_FV_V1_0_0 REFRESH", [], False),
            (
                "SELECT * FROM INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY",
                [{"TABLE_NAME": "FEATURE_VIEW_FV_V1_0_0"}],
                False,
            ),
        ]
    )
    fs = FeatureStore(session=sess, database="DB", schema="FEAT")
    fv = DummyFeatureView("FV", "1.0.0", True, "SELECT 1")
    fv.entities = [Entity(name="CUSTOMER", join_keys=["ID"])]
    fs.register_feature_view(fv)

    fs.suspend_feature_view("FV")
    fs.resume_feature_view("FV")
    fs.refresh_feature_view("FV")
    history = fs.get_feature_view_refresh_history("FV", limit=5)

    assert any("SUSPEND" in q for q in sess.queries)
    assert any("RESUME" in q for q in sess.queries)
    assert any("REFRESH" in q for q in sess.queries)
    assert history[0]["TABLE_NAME"] == "FEATURE_VIEW_FV_V1_0_0"


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
    fv1 = DummyFeatureView("FV", "1.0.0", False, "SELECT 1")
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
    fv2 = DummyFeatureView("FV", "2.0.0", False, "SELECT 1")
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
    fv3 = DummyFeatureView("FV", "3.0.0", False, "SELECT 1")
    with pytest.raises(FeatureStoreError):
        fs3.register_feature_view(fv3)

    # getters
    assert fs1.get_feature_view("FV").name == "FV"
    assert "FV" in fs1.list_feature_views()


def test_register_feature_view_applies_governance_policies():
    """Ensure governance configuration triggers tag and masking statements."""
    sess = StubSession(
        [
            ("CREATE SCHEMA IF NOT EXISTS", [], False),
            ("USE SCHEMA", [], False),
            ("DROP DYNAMIC TABLE IF EXISTS", [], False),
            ("CREATE OR REPLACE DYNAMIC TABLE", [], False),
            ("ALTER DYNAMIC TABLE DB.FEAT.FEATURE_VIEW_FV_V1_0_0 SET TAG", [], False),
            (
                "ALTER DYNAMIC TABLE DB.FEAT.FEATURE_VIEW_FV_V1_0_0 MODIFY COLUMN",
                [],
                False,
            ),
        ]
    )
    governance_config = {
        "feature_views": {
            "FV": {
                "tags": {"governance.data_classification": "HIGH"},
                "masking_policies": {"SENSITIVE_COL": "SECURE_SCHEMA.SENSITIVE_MASK"},
            }
        }
    }
    fs = FeatureStore(
        session=sess,
        database="DB",
        schema="FEAT",
        governance=governance_config,
    )
    fv = DummyFeatureView("FV", "1.0.0", True, "SELECT 1 AS SENSITIVE_COL")
    fv.entities = [Entity(name="CUSTOMER", join_keys=["ID"])]

    fs.register_feature_view(fv, overwrite=True)

    assert any("SET TAG" in q for q in sess.queries)
    assert any(
        'MODIFY COLUMN "SENSITIVE_COL" SET MASKING POLICY SECURE_SCHEMA.SENSITIVE_MASK'
        in q
        for q in sess.queries
    )
