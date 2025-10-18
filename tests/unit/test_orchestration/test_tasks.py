"""Unit tests for task orchestrator."""

from snowflake_ml_template.orchestration.tasks import TaskOrchestrator


class StubSQL:
    """Stub SQL."""

    def __init__(self, raise_exc=False):
        """Stub SQL."""
        self._raise = raise_exc

    def collect(self):
        """Collect binds."""
        if self._raise:
            raise RuntimeError("boom")
        return []


class StubSession:
    """Stub session."""

    def __init__(self, fail_pred=None):
        """Stub session."""
        self.queries = []
        self.fail_pred = fail_pred or (lambda q: False)

    def sql(self, query: str):
        """Stub SQL query."""
        self.queries.append(query)
        return StubSQL(self.fail_pred(query))


def test_create_task_with_schedule_success():
    """Test create task with schedule success."""
    sess = StubSession()
    orch = TaskOrchestrator(sess, "DB", "PIPE")
    ok = orch.create_task(name="t1", sql="SELECT 1", schedule="CRON", warehouse="WH")
    assert ok is True
    assert any("CREATE OR REPLACE TASK DB.PIPE.t1" in q for q in sess.queries)


def test_create_task_with_after_dependency_success():
    """Test create task with after dependency success."""
    sess = StubSession()
    orch = TaskOrchestrator(sess, "DB", "PIPE")
    ok = orch.create_task(name="t2", sql="SELECT 1", after=["a", "b"], warehouse="WH")
    assert ok is True
    assert any("AFTER a, b" in q for q in sess.queries)


def test_create_task_failure_returns_false():
    """Test create task failure returns false."""
    sess = StubSession(fail_pred=lambda q: True)
    orch = TaskOrchestrator(sess, "DB", "PIPE")
    assert orch.create_task(name="t3", sql="SELECT 1") is False


def test_resume_task_success_and_failure():
    """Test resume task success and failure."""
    sess_ok = StubSession()
    orch_ok = TaskOrchestrator(sess_ok, "DB", "PIPE")
    assert orch_ok.resume_task("t4") is True

    sess_fail = StubSession(fail_pred=lambda q: True)
    orch_fail = TaskOrchestrator(sess_fail, "DB", "PIPE")
    assert orch_fail.resume_task("t5") is False
