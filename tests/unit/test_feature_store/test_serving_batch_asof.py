"""Unit tests validating ASOF join SQL generation."""

from snowflake_ml_template.feature_store.serving.batch import BatchFeatureServer


class SpineStub:
    """Stub spine DataFrame exposing SQL text."""

    def __init__(self, name: str, columns, sql: str | None = None):
        """Stub init."""
        self.columns = list(columns)
        self._sql = sql or f"SELECT * FROM {name}"

    def to_sql(self):
        """Stub to_sql."""
        return self._sql


class FeatureStub:
    """Stub feature DataFrame returned by session.table."""

    def __init__(self, columns):
        """Stub init."""
        self.columns = list(columns)


class ResultStub:
    """Stub result returned by session.sql capturing generated SQL."""

    def __init__(self, sql: str):
        """Stub init."""
        self.sql = sql
        self.columns: list[str] = []

    def set_columns(self, cols):
        """Stub set_columns."""
        self.columns = list(cols)


class SessionStub:
    """Stub Snowpark session capturing executed SQL statements."""

    def __init__(self, tables):
        """Stub init."""
        self.tables = tables
        self.queries: list[str] = []

    def table(self, name):
        """Stub table."""
        if name not in self.tables:
            raise RuntimeError("missing table")
        return self.tables[name]

    def sql(self, query):
        """Stub sql."""
        self.queries.append(query)
        return ResultStub(query)


def test_get_features_asof_join_emits_native_sql():
    """Ensure ASOF join path produces native LEFT ASOF JOIN SQL."""
    spine = SpineStub("SPINE", ["ENTITY_ID", "TS"], "SELECT ENTITY_ID, TS FROM SPINE")
    fv = FeatureStub(["ENTITY_ID", "TS", "FEATURE_A"])
    sess = SessionStub({"DB.FEAT.FEATURE_VIEW_fv_asof": fv})
    bfs = BatchFeatureServer(sess, "DB", "FEAT")

    result = bfs.get_features(
        spine_df=spine,
        feature_views=["fv_asof"],
        spine_timestamp_col="TS",
        spine_entity_cols=["ENTITY_ID"],
        asof_tolerance="INTERVAL '5' MINUTE",
    )

    query = sess.queries[-1]
    assert "LEFT ASOF JOIN" in query
    assert "MATCH_CONDITION" in query
    assert "INTERVAL '5' MINUTE" in query
    assert 'SPINE_0."ENTITY_ID" = FV_0."ENTITY_ID"' in query
    assert 'FV_0."TS" <= SPINE_0."TS"' in query
    assert isinstance(result, ResultStub)
