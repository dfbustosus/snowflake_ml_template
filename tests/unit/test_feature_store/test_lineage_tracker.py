"""Unit tests for lineage tracking."""

from snowflake_ml_template.feature_store.versioning.lineage import LineageTracker


class StubSQL:
    """Stub SQL."""

    def __init__(self, behavior, binds):
        """Stub init."""
        self._behavior = behavior
        self._binds = binds

    def bind(self, *args):
        """Stub bind."""
        self._binds.extend(args)
        return self

    def collect(self):
        """Stub collect."""
        return self._behavior(self._binds)


class StubSession:
    """Stub session."""

    def __init__(self, routes):
        """Stub init."""
        # routes: list of (substring, behavior)
        self.routes = routes
        self.queries = []
        self.binds = []

    def sql(self, query: str):
        """Stub sql."""
        self.queries.append(query)

        def behavior(args):
            for substr, beh in self.routes:
                if substr in query:
                    return beh(args)
            return []

        return StubSQL(behavior, self.binds)


def test_init_creates_tables():
    """Test init creates tables."""
    sess = StubSession(
        [
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    assert any("LINEAGE_NODES" in q for q in sess.queries)
    assert any("LINEAGE_EDGES" in q for q in sess.queries)


def test_add_source_table_stores_node_and_returns():
    """Test add source table stores node and returns."""
    captured = {"stored": False}

    def merge_behavior(args):
        # args: [node.id, node.id, node.type.value, node.name, json, created_at]
        captured["stored"] = True
        return []

    sess = StubSession(
        [
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
            ("MERGE INTO", merge_behavior),
        ]
    )
    tracker = LineageTracker(sess, "DB", "FEAT")
    node = tracker.add_source_table("T1", "RAW.TBL")
    assert node.name == "RAW.TBL"
    assert captured["stored"] is True


def test_add_feature_view_lineage_creates_nodes_and_edges():
    """Test add feature view lineage creates nodes and edges."""
    events = []

    def merge_behavior(args):
        events.append(("merge", tuple(args)))
        return []

    def insert_edge_behavior(args):
        events.append(("insert_edge", tuple(args)))
        return []

    sess = StubSession(
        [
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
            ("MERGE INTO", merge_behavior),
            ("INSERT INTO", insert_edge_behavior),
        ]
    )
    tracker = LineageTracker(sess, "DB", "FEAT")
    tracker.add_feature_view_lineage("FV", ["SRC1", "SRC2"], transformation_logic="sql")
    # Expect at least one MERGE and two INSERTs
    assert any(ev[0] == "merge" for ev in events)
    assert sum(1 for ev in events if ev[0] == "insert_edge") >= 2


def test_get_upstream_and_downstream_lineage_mapping():
    """Test get upstream and downstream lineage mapping."""
    upstream_rows = [
        {
            "NODE_ID": "SRC1",
            "NODE_TYPE": "source_table",
            "NODE_NAME": "RAW.S1",
            "DEPTH": 1,
        },
        {
            "NODE_ID": "SRC2",
            "NODE_TYPE": "source_table",
            "NODE_NAME": "RAW.S2",
            "DEPTH": 1,
        },
    ]
    downstream_rows = [
        {"NODE_ID": "FV1", "NODE_TYPE": "feature_view", "NODE_NAME": "FV1", "DEPTH": 1},
    ]
    sess = StubSession(
        [
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
            ("WITH RECURSIVE upstream", lambda _: upstream_rows),
            ("WITH RECURSIVE downstream", lambda _: downstream_rows),
        ]
    )
    tracker = LineageTracker(sess, "DB", "FEAT")
    up = tracker.get_upstream_lineage("fv_FV")
    down = tracker.get_downstream_lineage("SRC1")
    assert up[0]["node_id"] == "SRC1" and down[0]["type"] == "feature_view"


def test_get_impact_analysis_groups_by_type(monkeypatch):
    """Test get impact analysis groups by type."""
    sess = StubSession(
        [
            ("CREATE TABLE IF NOT EXISTS", lambda _: []),
        ]
    )
    tracker = LineageTracker(sess, "DB", "FEAT")
    monkeypatch.setattr(
        LineageTracker,
        "get_downstream_lineage",
        lambda self, nid: [
            {"node_id": "fv1", "type": "feature_view", "name": "FV1", "depth": 1},
            {"node_id": "src1", "type": "source_table", "name": "S1", "depth": 1},
        ],
    )
    impact = tracker.get_impact_analysis("fv1")
    assert impact["total_impacted"] == 2
    assert set(impact["impact_by_type"].keys()) == {"feature_view", "source_table"}
