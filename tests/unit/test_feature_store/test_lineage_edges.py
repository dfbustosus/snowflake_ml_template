"""Unit tests for lineage edges."""

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

    def __init__(self):
        """Stub init."""
        self.queries = []
        self.binds = []
        self._insert_seen = set()

    def sql(self, query: str):
        """Stub sql."""
        self.queries.append(query)

        def behavior(args):
            """Stub behavior."""
            if query.strip().startswith("CREATE TABLE IF NOT EXISTS"):
                return []
            if query.strip().startswith("MERGE INTO"):
                return []
            if query.strip().startswith("INSERT INTO"):
                # args: [edge_id, source_id, target_id, relationship, metadata, created_at, edge_id]
                edge_id = args[0]
                # Simulate WHERE NOT EXISTS by ignoring duplicate edge_ids
                if edge_id in self._insert_seen:
                    return []
                self._insert_seen.add(edge_id)
                return []
            if query.strip().startswith("WITH RECURSIVE upstream"):
                # return empty lineage
                return []
            return []

        return StubSQL(behavior, self.binds)


def test_duplicate_edge_prevented_by_edge_id_condition():
    """Test duplicate edge prevented by edge id condition."""
    sess = StubSession()
    tracker = LineageTracker(sess, "DB", "SCH")
    # register source tables
    tracker.add_source_table("SRC1", "RAW.SRC1")
    # first add
    tracker.add_feature_view_lineage("fvx", ["SRC1"])
    # second add same lineage
    tracker.add_feature_view_lineage("fvx", ["SRC1"])

    insert_queries = [q for q in sess.queries if "INSERT INTO" in q]
    assert len(insert_queries) >= 2
    # Ensure both inserts used the same edge_id (first and second bind arg index 0)
    # binds order: [node merge binds..., then for first insert edge binds (7 items), then second insert edge binds]
    # Extract edge_ids by walking binds in steps of 7 starting after merges (we don't know exact count, so scan by query)
    edge_ids = []
    idx = 0
    # Re-simulate by matching calls in order: for each INSERT INTO occurrence, capture the next first bind
    for q in sess.queries:
        if "INSERT INTO" in q:
            edge_ids.append(sess.binds[idx])
            idx += 7
        elif "MERGE INTO" in q:
            idx += 6
        elif "CREATE TABLE IF NOT EXISTS" in q:
            # no binds
            pass
    assert len(edge_ids) >= 2
    assert edge_ids[0] == edge_ids[1]


def test_get_upstream_uses_max_depth_bind():
    """Test get upstream uses max depth bind."""
    sess = StubSession()
    tracker = LineageTracker(sess, "DB", "SCH")
    # call upstream with explicit depth
    out = tracker.get_upstream_lineage("fv_node", max_depth=2)
    # The last two binds should be ["fv_node", 2]
    assert "WITH RECURSIVE upstream" in " ".join(sess.queries)
    assert sess.binds[-2:] == ["fv_node", 2]
    assert out == []
