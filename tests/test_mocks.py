"""Utility tests that provide mocks and test fixtures used in other tests.

These helpers create small fake modules for the Snowflake ML SDK so other
tests can run without heavy third-party dependencies.
"""

import os
import sys
import types
from importlib import reload

from snowflake_ml_template.feature_store.api import FeatureStoreHelper
from snowflake_ml_template.models.registry import ModelRegistryHelper
from snowflake_ml_template.utils.session import create_session_from_env, set_query_tag


def _make_fake_featurestore_module():
    """Create a fake feature store module for testing."""
    mod = types.SimpleNamespace()

    class Entity:
        def __init__(self, name, join_keys):
            self.name = name
            self.join_keys = join_keys

    class FeatureView:
        def __init__(self, name, entities, feature_df, refresh_freq=None):
            self.name = name

    called = {"reg_entity": False, "reg_fv": False}

    class FeatureStore:
        def __init__(self, session=None):
            self.session = session

        def register_entity(self, entity):
            called["reg_entity"] = True

        def register_feature_view(self, feature_view, version=None):
            called["reg_fv"] = True

    mod.Entity = Entity
    mod.FeatureView = FeatureView
    mod.FeatureStore = FeatureStore
    mod._called = called
    return mod


def _make_fake_model_module():
    """Create a fake model module for testing."""
    mod = types.SimpleNamespace()

    class ModelRegistry:
        def __init__(self, session=None):
            self.session = session

        def log_model(self, model_name, model_path, version_name, metadata=None):
            return {
                "model_name": model_name,
                "version": version_name,
                "path": model_path,
            }

    mod.ModelRegistry = ModelRegistry
    return mod


def _make_fake_snowpark_module():
    """Create a fake snowpark module for testing."""

    class FakeBuilder:
        def __init__(self):
            self._configs = {}

        def configs(self, cfg):
            self._configs.update(cfg)
            return self

        def create(self):
            class DummySession:
                def __init__(self, configs):
                    self._configs = configs

                def sql(self, q):
                    class Q:
                        def __init__(self, q):
                            self.q = q

                        def collect(self):
                            return [(self.q,)]

                    return Q(q)

            return DummySession(self._configs)

    class Session:
        builder = FakeBuilder()

    return types.SimpleNamespace(Session=Session)


def test_feature_store_helper_register(monkeypatch):
    """Ensure FeatureStoreHelper.register_entity and register_feature_view run with mocks."""
    fake_mod = _make_fake_featurestore_module()
    snowflake_ml_fs_name = "snowflake.ml.feature_store"
    snowflake_ml_name = "snowflake.ml"
    snowflake_name = "snowflake"
    sys.modules.setdefault(snowflake_name, types.SimpleNamespace())
    sys.modules.setdefault(snowflake_ml_name, types.SimpleNamespace())
    sys.modules[snowflake_ml_fs_name] = types.SimpleNamespace(
        Entity=fake_mod.Entity,
        FeatureStore=fake_mod.FeatureStore,
        FeatureView=fake_mod.FeatureView,
    )

    # reload the api module to pick up mocked modules if necessary
    reload(sys.modules[__name__])

    helper = FeatureStoreHelper()
    session = object()
    helper.register_entity(session, "CUSTOMER", ["CUSTOMER_ID"])
    helper.register_feature_view(
        session, name="FV", feature_df=None, entities=[], refresh_freq=None
    )

    assert hasattr(helper, "register_entity")


def test_model_registry_log_and_promote(monkeypatch):
    """Test logging and promoting models in the model registry."""
    fake_model_mod = _make_fake_model_module()
    sys.modules.setdefault("snowflake", types.SimpleNamespace())
    sys.modules["snowflake.ml.model"] = types.SimpleNamespace(
        ModelRegistry=fake_model_mod.ModelRegistry
    )

    helper = ModelRegistryHelper()
    mv = helper.log_model(
        session=None,
        model_name="CHURN",
        model_file="@ML_MODELS_STAGE/churn/v1/model.joblib",
        version_name="v1",
    )
    assert mv["model_name"] == "CHURN"
    sql = helper.promote_model(None, "ML_TEST_DB", "ML_PROD_DB", "CHURN", "v1")
    assert "CREATE MODEL" in sql


def test_utils_set_query_tag_and_create_session(monkeypatch):
    """Test setting query tags and creating sessions."""

    class DummySession:
        def __init__(self):
            self.last = None

        def sql(self, q):
            self.last = q

            class Q:
                def collect(self_inner):
                    return [q]

            return Q()

    ds = DummySession()
    set_query_tag(ds, "TEST_TAG")
    assert "QUERY_TAG = 'TEST_TAG'" in ds.last

    fake_sp = _make_fake_snowpark_module()
    sys.modules["snowflake.snowpark"] = fake_sp
    os.environ["SNOWFLAKE_ACCOUNT"] = "acct"
    os.environ["SNOWFLAKE_USER"] = "user"
    os.environ["SNOWFLAKE_PASSWORD"] = "pw"
    os.environ["SNOWFLAKE_ROLE"] = "role"
    os.environ["SNOWFLAKE_WAREHOUSE"] = "wh"
    os.environ["SNOWFLAKE_DATABASE"] = "db"
    os.environ["SNOWFLAKE_SCHEMA"] = "schema"

    sess = create_session_from_env()
    assert hasattr(sess, "_configs")
    os.environ["SNOWFLAKE_SCHEMA"] = "schema"
