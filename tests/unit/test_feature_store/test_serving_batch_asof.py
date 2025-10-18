"""Unit tests for batch feature server."""

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

    def desc(self):
        """Stub desc."""
        return ("desc", self.name)


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


class SessionStub:
    """Stub session."""

    def __init__(self, tables):
        """Stub session."""
        self.tables = tables

    def table(self, name):
        """Stub table."""
        if name not in self.tables:
            raise RuntimeError("missing table")
        return self.tables[name]


def test_get_features_asof_join_path_records_expected_ops(monkeypatch):
    """Test get features asof join path records expected ops."""
    # Monkeypatch Snowpark Window and row_number to lightweight stubs to avoid type checks
    import sys
    import types

    class _SpecStub:
        """Stub spec."""

        def order_by(self, *args, **kwargs):
            """Stub order by."""
            return self

    class _WindowStub:
        """Stub window."""

        @staticmethod
        def partition_by(*cols, **kwargs):
            """Stub partition by."""
            return _SpecStub()

    class _RowNumber:
        """Stub row number."""

        def over(self, spec):
            """Stub over."""
            return ("row_number_over", spec)

    win_module = types.SimpleNamespace(Window=_WindowStub)
    monkeypatch.setitem(sys.modules, "snowflake.snowpark.window", win_module)
    try:
        import snowflake.snowpark.functions as sfuncs

        monkeypatch.setattr(sfuncs, "row_number", lambda: _RowNumber(), raising=False)
    except Exception:
        pass
    spine = DFStub("spine", ["ENTITY_ID", "TS"])
    # feature df includes entity, timestamp, and a feature column prefixed with FEATURE_
    fv = DFStub("fv_table", ["ENTITY_ID", "TS", "FEATURE_A"])
    sess = SessionStub({"DB.FEAT.FEATURE_VIEW_fv_asof": fv})
    bfs = BatchFeatureServer(sess, "DB", "FEAT")

    out = bfs.get_features(
        spine,
        feature_views=["fv_asof"],
        spine_timestamp_col="TS",
        spine_entity_cols=["ENTITY_ID"],
    )
    assert any(op[0] == "join" for op in out.ops)
    assert any(op[0] == "with_column" and op[1] == "_row_num" for op in out.ops)
    assert any(op[0] == "filter" for op in out.ops)
    assert any(op[0] == "drop" and op[1] == "_row_num" for op in out.ops)
