"""Integration-style tests for governance and monitoring flows using stubs."""

from unittest.mock import Mock

import pytest

from snowflake_ml_template.feature_store.core.entity import Entity
from snowflake_ml_template.feature_store.core.feature_view import FeatureView
from snowflake_ml_template.feature_store.core.store import FeatureStore
from snowflake_ml_template.feature_store.monitoring.quality import (
    FeatureQualityMonitor,
    QualityMetrics,
)


class FeatureStoreSessionStub:
    """Stub Snowpark session capturing SQL for governance tests."""

    def __init__(self, tables):
        """Initialize the stub session."""
        self.tables = tables
        self.queries = []
        self.feature_store = Mock()

    def sql(self, query):
        """Execute a SQL query."""
        self.queries.append(query)

        class _Result:
            def collect(self_inner):
                return []

            def bind(self_inner, *args, **kwargs):
                return self_inner

        return _Result()

    def table(self, name):
        """Return a table stub."""
        stub = Mock()
        stub.columns = self.tables.get(name, [])
        return stub


class MonitoringSessionStub:
    """Stub session capturing SQL for monitoring persistence/tests."""

    def __init__(self):
        """Initialize the stub session."""
        self.queries = []

    def sql(self, query):
        """Execute a SQL query."""
        self.queries.append(query)

        class _Result:
            def collect(self_inner):
                """Collect query results."""
                return []

        return _Result()


@pytest.fixture
def governance_session():
    """Return a stub session for governance tests."""
    return FeatureStoreSessionStub(
        {"DB.FEAT.FEATURE_VIEW_FV_V1_0_0": ["ID", "SENSITIVE_COL"]}
    )


@pytest.fixture
def monitoring_session():
    """Return a stub session for monitoring tests."""
    return MonitoringSessionStub()


def test_feature_store_governance_applies_tags_and_policies(governance_session):
    """Ensure governance config produces tagging and masking statements."""
    governance = {
        "feature_views": {
            "FV": {
                "tags": {"governance.data_classification": "RESTRICTED"},
                "masking_policies": {"SENSITIVE_COL": "SECURE.MASK_POLICY"},
            }
        }
    }
    fs = FeatureStore(governance_session, "DB", "FEAT", governance=governance)

    feature_df = Mock()
    feature_df.columns = ["ID", "SENSITIVE_COL"]
    feature_df.to_sql.return_value = "SELECT ID, SENSITIVE_COL FROM SOURCE"
    fv = FeatureView(
        name="FV",
        entities=[Entity(name="CUSTOMER", join_keys=["ID"])],
        feature_df=feature_df,
        refresh_freq="1 hour",
    )

    fs.register_feature_view(fv, overwrite=True)

    assert any("SET TAG" in q for q in governance_session.queries)
    assert any(
        "MASKING POLICY SECURE.MASK_POLICY" in q for q in governance_session.queries
    )


def test_quality_monitor_records_metrics_and_task(monitoring_session):
    """Verify quality monitoring persistence and task creation statements."""
    monitor = FeatureQualityMonitor(
        monitoring_session,
        database="DB",
        schema="MONITOR",
        table="QUALITY_EVENTS",
    )

    metrics = QualityMetrics(
        feature_name="TOTAL_AMOUNT",
        total_rows=1000,
        null_count=5,
        null_rate=0.005,
        unique_count=900,
        mean=42.1,
        std=3.2,
        min=10.0,
        max=80.0,
        quality_score=0.92,
    )

    monitor.record_metrics(metrics, feature_view="SALES_FV", run_id="2025-10-18")
    monitor.create_quality_task(
        task_name="DB.MONITOR.QUALITY_TASK",
        warehouse="MONITOR_WH",
        schedule="USING CRON */15 * * * * UTC",
        procedure_call="CALL DB.MONITOR.RUN_QUALITY()",
    )

    assert any(
        "INSERT INTO DB.MONITOR.QUALITY_EVENTS" in q for q in monitoring_session.queries
    )
    assert any(
        "CREATE OR REPLACE TASK DB.MONITOR.QUALITY_TASK" in q
        for q in monitoring_session.queries
    )
