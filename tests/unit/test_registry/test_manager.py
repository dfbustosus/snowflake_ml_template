"""Unit tests for model registry."""

import pytest

from snowflake_ml_template.core.exceptions.errors import VersionNotFoundError
from snowflake_ml_template.registry.manager import (
    ModelRegistry,
    ModelStage,
    ModelVersion,
)


class StubSQL:
    """Stub SQL."""

    def __init__(self, behavior, binds):
        """Stub SQL."""
        self._behavior = behavior
        self._binds = binds

    def bind(self, *args):
        """Bind args."""
        self._binds.extend(args)
        return self

    def collect(self):
        """Collect binds."""
        return self._behavior(self._binds)


class StubSession:
    """Stub session."""

    def __init__(self, routes):
        """Stub session."""
        # routes: list of tuples (match_substring, behavior)
        self.routes = routes
        self.queries = []
        self.binds = []

    def sql(self, query: str):
        """Stub SQL query."""
        self.queries.append(query)

        def behavior(args):
            for substr, beh in self.routes:
                if substr in query:
                    return beh(args)
            return []

        return StubSQL(behavior, self.binds)


def test_registry_init_and_tables_created():
    """Test registry init and tables created."""
    # any sql returns []
    sess = StubSession(
        routes=[
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    # Instantiate registry to trigger table creation
    _ = ModelRegistry(sess, "DB", "MODELS")
    assert any(
        "CREATE TABLE IF NOT EXISTS DB.MODELS.MODEL_REGISTRY" in q for q in sess.queries
    )
    assert any(
        "CREATE TABLE IF NOT EXISTS DB.MODELS.MODEL_VERSIONS" in q for q in sess.queries
    )


def test_register_model_inserts_registry_and_version(monkeypatch):
    """Test register model inserts registry and version."""

    def route_behavior(args):
        return []

    sess = StubSession(
        routes=[
            ("MERGE INTO", lambda _: []),
            ("INSERT INTO", lambda _: []),
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    reg = ModelRegistry(sess, "DB", "MODELS")
    mv = reg.register_model(
        model_name="fraud",
        version="1.2.3",
        stage=ModelStage.DEV,
        artifact_path="@stage/model",
        framework="xgboost",
        metrics={"auc": 0.9},
        signature={"in": "x"},
        dependencies={"pkg": "1.0"},
        created_by="me",
    )
    assert isinstance(mv, ModelVersion)
    assert mv.version == "1.2.3" and mv.stage == ModelStage.DEV


def test_set_default_version_updates_registry():
    """Test set default version updates registry."""
    sess = StubSession(
        routes=[
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
            ("UPDATE DB.MODELS.MODEL_REGISTRY", lambda _: []),
        ]
    )
    reg = ModelRegistry(sess, "DB", "MODELS")
    reg.set_default_version("fraud", "2.0.0")
    assert any("UPDATE DB.MODELS.MODEL_REGISTRY" in q for q in sess.queries)


def test_version_exists_true_false():
    """Test version exists true false."""
    sess_true = StubSession(
        routes=[
            ("SELECT COUNT(*) as count", lambda _: [{"count": 1}]),
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    reg_true = ModelRegistry(sess_true, "DB", "MODELS")
    assert reg_true.version_exists("fraud", "1.0.0") is True

    sess_false = StubSession(
        routes=[
            ("SELECT COUNT(*) as count", lambda _: [{"count": 0}]),
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    reg_false = ModelRegistry(sess_false, "DB", "MODELS")
    assert reg_false.version_exists("fraud", "1.0.0") is False


def test_get_version_success_and_not_found():
    """Test get version success and not found."""
    row = {
        "MODEL_NAME": "fraud",
        "VERSION": "1.0.0",
        "STAGE": "dev",
        "ARTIFACT_PATH": "@stage/model",
        "FRAMEWORK": "xgboost",
        "METRICS": {},
        "SIGNATURE": None,
        "DEPENDENCIES": None,
        "CREATED_AT": None,
        "CREATED_BY": "me",
    }
    sess_ok = StubSession(
        routes=[
            ("SELECT * FROM DB.MODELS.MODEL_VERSIONS", lambda _: [row]),
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    reg_ok = ModelRegistry(sess_ok, "DB", "MODELS")
    mv = reg_ok.get_version("fraud", "1.0.0")
    assert mv.model_name == "fraud" and mv.version == "1.0.0"

    sess_nf = StubSession(
        routes=[
            ("SELECT * FROM DB.MODELS.MODEL_VERSIONS", lambda _: []),
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    reg_nf = ModelRegistry(sess_nf, "DB", "MODELS")
    with pytest.raises(VersionNotFoundError):
        reg_nf.get_version("fraud", "1.0.0")


def test_list_versions_with_and_without_stage():
    """Test list versions with and without stage."""
    sess = StubSession(
        routes=[
            (
                "SELECT DISTINCT version",
                lambda _: [{"VERSION": "1.0.0"}, {"VERSION": "2.0.0"}],
            ),
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    reg = ModelRegistry(sess, "DB", "MODELS")
    assert reg.list_versions("fraud") == ["1.0.0", "2.0.0"]
    # stage filter path
    reg.list_versions("fraud", ModelStage.DEV)
    assert any("AND stage = ?" in q for q in sess.queries)


def test_promote_model_flows(monkeypatch):
    """Test promote model flows."""
    # monkeypatch version_exists and get_version to avoid heavy SQL
    sess = StubSession(routes=[("CREATE TABLE IF NOT EXISTS", lambda _: [])])
    reg = ModelRegistry(sess, "DB", "MODELS")

    # not exists path
    monkeypatch.setattr(ModelRegistry, "version_exists", lambda self, m, v: False)
    with pytest.raises(VersionNotFoundError):
        reg.promote_model("fraud", "1.0.0", ModelStage.TEST)

    # exists path: mock get_version and register_model
    class MV:
        pass

    mv = ModelVersion(
        model_name="fraud",
        version="1.0.0",
        stage=ModelStage.DEV,
        artifact_path="@",
        framework="xgboost",
    )
    monkeypatch.setattr(ModelRegistry, "version_exists", lambda self, m, v: True)
    monkeypatch.setattr(ModelRegistry, "get_version", lambda self, m, v: mv)
    called = {"registered": False}

    def fake_register(**kwargs):
        called["registered"] = True
        return mv

    monkeypatch.setattr(
        ModelRegistry, "register_model", lambda self, **kwargs: fake_register(**kwargs)
    )
    reg.promote_model("fraud", "1.0.0", ModelStage.PROD)
    assert called["registered"] is True
