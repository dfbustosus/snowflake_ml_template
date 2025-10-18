"""Unit tests for stages."""

import pytest

from snowflake_ml_template.core.exceptions import ConfigurationError
from snowflake_ml_template.infrastructure.provisioning.stages import StageProvisioner


class SQLStub:
    """Stub SQL."""

    def __init__(self, behavior):
        """Stub SQL."""
        self._behavior = behavior

    def collect(self):
        """Stub collect."""
        return self._behavior()


class SessionStub:
    """Stub session."""

    def __init__(self, behavior_map):
        """Stub session."""
        # behavior_map: list of (substring, callable)
        self.behavior_map = behavior_map
        self.queries = []

    def sql(self, query: str):
        """Stub SQL query."""
        self.queries.append(query)
        for substr, beh in self.behavior_map:
            if substr in query:
                return SQLStub(beh)
        return SQLStub(lambda: [])


def test_create_stage_minimal_success():
    """Test create stage minimal success."""
    sess = SessionStub([("CREATE STAGE IF NOT EXISTS", lambda: [])])
    prov = StageProvisioner(sess)
    ok = prov.create_stage(name="STG", database="DB", schema="SCH")
    assert ok is True
    assert any('CREATE STAGE IF NOT EXISTS "DB"."SCH"."STG"' in q for q in sess.queries)


def test_create_stage_with_url_integration_file_format_and_comment():
    """Test create stage with url integration file format and comment."""

    def beh():
        return []

    sess = SessionStub([("CREATE STAGE IF NOT EXISTS", beh)])
    prov = StageProvisioner(sess)
    ok = prov.create_stage(
        name="STG2",
        database="DB",
        schema="SCH",
        url="s3://bucket/path",
        storage_integration="INTG",
        file_format="FMT_JSON",
        comment="my stage",
    )
    assert ok is True
    q = " ".join(sess.queries)
    assert "URL = 's3://bucket/path'" in q
    assert 'STORAGE_INTEGRATION = "INTG"' in q
    assert 'FILE_FORMAT = "FMT_JSON"' in q
    assert "COMMENT = 'my stage'" in q


def test_create_stage_failure_raises_configuration_error():
    """Test create stage failure raises configuration error."""
    sess = SessionStub(
        [
            (
                "CREATE STAGE IF NOT EXISTS",
                lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            )
        ]
    )
    prov = StageProvisioner(sess)
    with pytest.raises(ConfigurationError):
        prov.create_stage(name="STG", database="DB", schema="SCH")


def test_stage_exists_true_false_and_exception():
    """Test stage exists true false and exception."""
    # True case
    sess_true = SessionStub([("SHOW STAGES LIKE", lambda: [{"name": "STG"}])])
    prov_true = StageProvisioner(sess_true)
    assert prov_true.stage_exists("STG", "DB", "SCH") is True

    # False case
    sess_false = SessionStub([("SHOW STAGES LIKE", lambda: [])])
    prov_false = StageProvisioner(sess_false)
    assert prov_false.stage_exists("STG", "DB", "SCH") is False

    # Exception -> False
    sess_err = SessionStub(
        [("SHOW STAGES LIKE", lambda: (_ for _ in ()).throw(RuntimeError("err")))]
    )
    prov_err = StageProvisioner(sess_err)
    assert prov_err.stage_exists("STG", "DB", "SCH") is False
