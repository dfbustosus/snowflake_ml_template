"""Unit tests for batch feature server."""
import pytest

from snowflake_ml_template.core.exceptions.errors import FeatureStoreError
from snowflake_ml_template.feature_store.serving.batch import BatchFeatureServer


class ColumnStub:
    """Stub column."""

    def __init__(self, name):
        """Stub init."""
        self.name = name

    def __eq__(self, other):
        """Stub eq."""
        return ("eq", self.name, getattr(other, "name", other))

    def __le__(self, other):
        """Stub le."""
        return ("le", self.name, getattr(other, "name", other))


class DFStub:
    """Stub dataframe."""

    def __init__(self, name, columns):
        """Stub init."""
        self._name = name
        self.columns = columns[:]
        self.ops = []

    def __getitem__(self, key):
        """Stub get item."""
        return ColumnStub(key)

    def join(self, other, condition, join_type="left"):
        """Stub join."""
        self.ops.append(
            ("join", other._name if hasattr(other, "_name") else other, join_type)
        )
        return self

    def with_column(self, name, expr):
        """Stub with column."""
        self.ops.append(("with_column", name))
        return self

    def filter(self, predicate):
        """Stub filter."""
        self.ops.append(("filter", predicate))
        return self

    def drop(self, col):
        """Stub drop."""
        self.ops.append(("drop", col))
        return self

    def select(self, cols):
        """Stub select."""
        # keep only requested columns
        self.columns = list(cols)
        self.ops.append(("select", tuple(cols)))
        return self


class SessionStub:
    """Stub session."""

    def __init__(self, tables):
        """Stub init."""
        # tables: dict full_name -> DFStub
        self.tables = tables

    def table(self, name):
        """Stub table."""
        if name not in self.tables:
            raise RuntimeError("missing table")
        return self.tables[name]


def test_get_features_simple_join_and_logging():
    """Test get features simple join and logging."""
    spine = DFStub("spine", ["ID", "TS", "LABEL"])
    fv = DFStub("fv_table", ["ID", "F1"])  # no timestamp col, triggers simple join path
    sess = SessionStub({"DB.FEAT.FEATURE_VIEW_fv1": fv})
    bfs = BatchFeatureServer(sess, "DB", "FEAT")
    out = bfs.get_features(
        spine, feature_views=["fv1"], spine_timestamp_col=None, spine_entity_cols=["ID"]
    )
    assert isinstance(out, DFStub)
    assert any(op[0] == "join" for op in out.ops)


def test_get_features_missing_table_raises():
    """Test get features missing table raises."""
    spine = DFStub("spine", ["ID"])
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
    """Test generate training dataset selects and excludes."""
    spine = DFStub("spine", ["ID", "TS", "LABEL"])
    fv = DFStub("fv_table", ["ID", "F1"])
    sess = SessionStub({"DB.FEAT.FEATURE_VIEW_fv1": fv})
    bfs = BatchFeatureServer(sess, "DB", "FEAT")
    ds = bfs.generate_training_dataset(
        spine,
        ["fv1"],
        label_col="LABEL",
        spine_timestamp_col=None,
        exclude_cols=["ID", "TS"],
    )
    assert "LABEL" in ds.columns and "ID" not in ds.columns and "TS" not in ds.columns
