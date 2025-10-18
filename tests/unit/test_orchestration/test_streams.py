"""Unit tests for stream processor."""

from snowflake_ml_template.orchestration.streams import StreamProcessor


class StubSQL:
    """Stub SQL."""

    def __init__(self, rows=None, raise_exc=False):
        """Stub SQL."""
        self._rows = rows or []
        self._raise = raise_exc

    def collect(self):
        """Stub collect."""
        if self._raise:
            raise RuntimeError("boom")
        return self._rows


class StubSession:
    """Stub session."""

    def __init__(self, routes):
        """Stub session."""
        # routes: list of (substring, rows, raise_exc)
        self.routes = routes
        self.queries = []

    def sql(self, query: str):
        """Stub SQL query."""
        self.queries.append(query)
        for substr, rows, raise_exc in self.routes:
            if substr in query:
                return StubSQL(rows, raise_exc)
        return StubSQL([])


def test_create_stream_success():
    """Test create stream success."""
    sess = StubSession([("CREATE OR REPLACE STREAM", [], False)])
    sp = StreamProcessor(sess, "DB", "RAW")
    assert sp.create_stream("s", "t") is True


def test_create_stream_failure():
    """Test create stream failure."""
    sess = StubSession([("CREATE OR REPLACE STREAM", [], True)])
    sp = StreamProcessor(sess, "DB", "RAW")
    assert sp.create_stream("s", "t") is False


def test_has_data_true_false_and_exception():
    """Test has data true false and exception."""
    # true
    sess_true = StubSession([("SYSTEM$STREAM_HAS_DATA", [(True,)], False)])
    sp_true = StreamProcessor(sess_true, "DB", "RAW")
    assert sp_true.has_data("s") is True

    # false (empty result)
    sess_false = StubSession([("SYSTEM$STREAM_HAS_DATA", [], False)])
    sp_false = StreamProcessor(sess_false, "DB", "RAW")
    assert sp_false.has_data("s") is False

    # exception
    sess_err = StubSession([("SYSTEM$STREAM_HAS_DATA", [], True)])
    sp_err = StreamProcessor(sess_err, "DB", "RAW")
    assert sp_err.has_data("s") is False
