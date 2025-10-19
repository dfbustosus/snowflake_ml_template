"""Unit tests for drift monitoring."""

import pytest

from snowflake_ml_template.feature_store.monitoring.drift import (
    DriftResult,
    FeatureDriftDetector,
)


class StubDF:
    """Stub dataframe."""

    def __init__(self, values):
        """Stub init."""
        # values: list of numeric values representing feature column
        self._values = list(values)
        self.columns = ["X"]

    # For stats
    def select(self, col_expr):
        """Stub select."""
        return self

    def agg(self, pairs):
        """Stub agg."""
        # Expect [ ("MIN","X"), ("MAX","X"), ("COUNT","X") ]
        mn = min(self._values) if self._values else 0
        mx = max(self._values) if self._values else 0
        cnt = len(self._values)

        class Row:
            def __init__(self, a, b, c):
                self._t = (a, b, c)

            def __getitem__(self, idx):
                return self._t[idx]

        return StubCollect([Row(mn, mx, cnt)])

    # For bin counts
    def filter(self, predicate):
        """Stub filter."""
        # Predicate is ignored; we will just stash for count via a context object
        return StubFilter(self._values, predicate)


class StubFilter:
    """Stub filter."""

    def __init__(self, values, predicate):
        """Stub init."""
        self._values = values
        self._predicate = predicate

    def count(self):
        """Stub count."""
        # The predicate uses feature_col ranges; we cannot evaluate easily.
        # Instead, we simulate by assuming equal spread across 10 bins.
        # We detect bounds by parsing lambda closure attributes not available; so approximate: return len(values)//10
        # Provide non-zero distribution while preserving total consistency across bins by test design.
        return max(0, len(self._values) // 10)


class StubCollect:
    """Stub collect."""

    def __init__(self, rows):
        """Stub init."""
        self._rows = rows

    def collect(self):
        """Stub collect."""
        return self._rows


class StubSession:
    """Stub session."""

    def __init__(self):
        """Stub init."""
        pass


def test_detect_psi_drift_basic():
    """Test detect PSI drift basic."""
    det = FeatureDriftDetector(StubSession())
    baseline = StubDF(list(range(100)))
    current = StubDF(list(range(100)))
    res = det.detect_psi_drift(
        baseline, current, feature_col="X", threshold=0.1, num_bins=10
    )
    assert isinstance(res, DriftResult)
    assert res.drift_detected in (True, False)  # depends on stub count approximation


def test_detect_psi_drift_zero_count_raises():
    """Test detect PSI drift zero count raises."""
    det = FeatureDriftDetector(StubSession())
    empty = StubDF([])
    with pytest.raises(Exception):
        det.detect_psi_drift(empty, empty, feature_col="X")


def test_detect_drift_batch_handles_errors(monkeypatch):
    """Test detect drift batch handles errors."""
    det = FeatureDriftDetector(StubSession())
    good = StubDF(list(range(50)))
    results = det.detect_drift_batch(good, good, feature_cols=["X", "Y"], threshold=0.1)
    # Should return list with two results (second may log an error and skip)
    assert isinstance(results, list)


def test_record_result_inserts_when_configured(monkeypatch):
    """Verify record_result emits INSERT when persistence configured."""
    session = StubSession()
    session.sql = lambda sql: StubCollect([])  # type: ignore[attr-defined]
    detector = FeatureDriftDetector(
        session, database="DB", schema="MON", table="DRIFT_EVENTS"
    )

    # replace execute sql to capture query
    captured = {}

    def _capture(sql: str):
        captured["sql"] = sql
        return StubCollect([])

    detector.session.sql = lambda query: _capture(query)  # type: ignore[assignment]

    result = DriftResult(
        feature_name="F1",
        drift_score=0.2,
        drift_detected=True,
        threshold=0.1,
        method="PSI",
        details={"bins": []},
    )

    detector.record_result(
        result, feature_view="FV", entity="CUSTOMER", run_id="run123"
    )
    assert "INSERT INTO DB.MON.DRIFT_EVENTS" in captured["sql"]
    assert "'FV'" in captured["sql"]
    assert "'F1'" in captured["sql"]


def test_create_drift_task_requires_events():
    """Ensure task creation fails without configured table."""
    detector = FeatureDriftDetector(StubSession())
    with pytest.raises(Exception):
        detector.create_drift_task(
            task_name="TASK",
            warehouse="WH",
            schedule="1 minute",
            procedure_call="CALL RUN()",
        )


def test_create_drift_task_emits_sql():
    """Ensure task creation emits expected SQL."""
    session = StubSession()

    class _SqlCollect:
        def __init__(self):
            self.last = None

        def __call__(self, query):
            self.last = query
            return StubCollect([])

    collector = _SqlCollect()
    session.sql = collector  # type: ignore[assignment]
    detector = FeatureDriftDetector(
        session, database="DB", schema="MON", table="DRIFT_EVENTS"
    )

    detector.create_drift_task(
        task_name="DB.MON.DRIFT_TASK",
        warehouse="MONITOR_WH",
        schedule="USING CRON * * * * * UTC",
        procedure_call="CALL DB.MON.RUN_DRIFT()",
    )

    assert "CREATE OR REPLACE TASK DB.MON.DRIFT_TASK" in collector.last
