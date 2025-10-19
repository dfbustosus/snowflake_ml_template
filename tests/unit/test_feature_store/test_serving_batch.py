"""Unit tests for batch feature server."""

import pytest

from snowflake_ml_template.core.exceptions.errors import FeatureStoreError
from snowflake_ml_template.feature_store.serving.batch import BatchFeatureServer


class SpineDFStub:
    """Stub spine DataFrame exposing SQL text."""

    def __init__(self, name: str, columns, sql: str | None = None):
        """Stub init."""
        self._name = name
        self.columns = list(columns)
        self._sql = sql or f"SELECT * FROM {name}"

    def to_sql(self):
        """Stub to_sql."""
        return self._sql


class FeatureDFStub:
    """Stub feature view DataFrame returned by session.table."""

    def __init__(self, name: str, columns):
        """Stub init."""
        self._name = name
        self.columns = list(columns)


class ResultDFStub:
    """Stub result DataFrame produced by session.sql."""

    def __init__(self, query: str):
        """Stub init."""
        self.query = query
        self.columns: list[str] = []
        self.select_calls: list[tuple[str, ...]] = []

    def set_columns(self, columns):
        """Stub set_columns."""
        self.columns = list(columns)

    def select(self, cols):
        """Stub select."""
        cols = tuple(cols)
        self.select_calls.append(cols)
        self.columns = list(cols)
        return self


class SessionStub:
    """Stub Snowpark session capturing SQL text."""

    def __init__(self, tables):
        """Stub init."""
        self.tables = tables
        self.queries: list[str] = []
        self.result_history: list[ResultDFStub] = []

    def table(self, name):
        """Stub table."""
        if name not in self.tables:
            raise RuntimeError("missing table")
        return self.tables[name]

    def sql(self, query):
        """Stub sql."""
        self.queries.append(query)
        result = ResultDFStub(query)
        self.result_history.append(result)
        return result


def test_get_features_simple_join_and_logging():
    """Test get_features emits LEFT JOIN SQL and tracks columns."""
    spine = SpineDFStub(
        "SPINE_SOURCE", ["ID", "TS", "LABEL"], "SELECT ID, TS, LABEL FROM SPINE_SOURCE"
    )
    fv = FeatureDFStub("FV_TABLE", ["ID", "F1"])
    sess = SessionStub({"DB.FEAT.FEATURE_VIEW_fv1": fv})
    bfs = BatchFeatureServer(sess, "DB", "FEAT")

    result = bfs.get_features(
        spine,
        feature_views=["fv1"],
        spine_timestamp_col=None,
        spine_entity_cols=["ID"],
    )

    assert isinstance(result, ResultDFStub)
    query = sess.queries[-1]
    assert "LEFT JOIN" in query
    assert "FEATURE_VIEW_fv1" in query
    assert 'SPINE_0."ID" = FV_0."ID"' in query
    assert result.columns == ["ID", "TS", "LABEL", "F1"]


def test_get_features_missing_table_raises():
    """Test get_features raises FeatureStoreError if feature view table missing."""
    spine = SpineDFStub("SPINE", ["ID"], "SELECT ID FROM SPINE")
    sess = SessionStub({})
    bfs = BatchFeatureServer(sess, "DB", "FEAT")
    with pytest.raises(FeatureStoreError):
        bfs.get_features(
            spine,
            feature_views=["does_not_exist"],
            spine_timestamp_col=None,
            spine_entity_cols=["ID"],
        )


def test_generate_training_dataset_selects_and_excludes():
    """Test training dataset projection excludes requested columns."""
    spine = SpineDFStub(
        "SPINE", ["ID", "TS", "LABEL"], "SELECT ID, TS, LABEL FROM SPINE"
    )
    fv = FeatureDFStub("FV", ["ID", "F1"])
    sess = SessionStub({"DB.FEAT.FEATURE_VIEW_fv1": fv})
    bfs = BatchFeatureServer(sess, "DB", "FEAT")

    dataset = bfs.generate_training_dataset(
        spine_df=spine,
        feature_views=["fv1"],
        label_col="LABEL",
        spine_timestamp_col="TS",
        spine_entity_cols=["ID"],
        exclude_cols=["ID", "TS"],
    )

    assert isinstance(dataset, ResultDFStub)
    query = sess.queries[-1]
    assert "LEFT ASOF JOIN" in query
    assert "MATCH_CONDITION" in query
    assert dataset.select_calls[0] == ("LABEL", "F1")
    assert dataset.columns == ["LABEL", "F1"]


def test_generate_inference_dataset_drops_entities_and_timestamp():
    """Inference dataset should exclude entity and timestamp columns."""
    spine = SpineDFStub("SPINE", ["ID", "TS"], "SELECT ID, TS FROM SPINE")
    fv = FeatureDFStub("FV", ["ID", "TS", "F1", "F2"])
    sess = SessionStub({"DB.FEAT.FEATURE_VIEW_fv1": fv})
    bfs = BatchFeatureServer(sess, "DB", "FEAT")

    dataset = bfs.generate_inference_dataset(
        spine_df=spine,
        feature_views=["fv1"],
        spine_timestamp_col="TS",
        spine_entity_cols=["ID"],
        exclude_cols=["TS"],
        asof_tolerance="INTERVAL '1' DAY",
    )

    query = sess.queries[-1]
    assert "LEFT ASOF JOIN" in query
    assert "INTERVAL '1' DAY" in query
    assert dataset.select_calls[0] == ("F1", "F2")
    assert dataset.columns == ["F1", "F2"]
