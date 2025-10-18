"""Unit tests for ingestion strategies failures."""

from snowflake_ml_template.core.base.ingestion import (
    DataSource,
    IngestionConfig,
    IngestionMethod,
    SourceType,
)
from snowflake_ml_template.ingestion.strategies.copy_into import CopyIntoStrategy
from snowflake_ml_template.ingestion.strategies.snowpipe import SnowpipeStrategy


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

    def __init__(self, routes):
        """Stub session."""
        # routes: list of (substring, callable)
        self.routes = routes

    def sql(self, query: str):
        """Stub SQL query."""
        for substr, beh in self.routes:
            if substr in query:
                return SQLStub(beh)
        return SQLStub(lambda: [])


def test_copy_into_ingest_failure_returns_failed_result():
    """Test copy into ingest failure returns failed result."""
    config = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/data", file_format="PARQUET"
        ),
        target_table="TABLE1",
        target_database="RAW_DATA",
        target_schema="PUBLIC",
        warehouse="TEST_WH",
        method=IngestionMethod.COPY_INTO,
    )
    strategy = CopyIntoStrategy(config)
    sess = SessionStub(
        [("COPY INTO", lambda: (_ for _ in ()).throw(RuntimeError("boom")))]
    )
    strategy.set_session(sess)
    result = strategy.execute_ingestion(
        config.source,
        "RAW_DATA.PUBLIC.TABLE1",
    )
    assert result.status == "failed"
    assert "boom" in (result.error or "")


def test_snowpipe_ingest_failure_returns_failed_result():
    """Test snowpipe ingest failure returns failed result."""
    config = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/stream", file_format="JSON"
        ),
        target_table="TABLE2",
        target_database="RAW_DATA",
        target_schema="PUBLIC",
        warehouse="TEST_WH",
        method=IngestionMethod.SNOWPIPE,
    )
    strategy = SnowpipeStrategy(config)
    sess = SessionStub(
        [
            (
                "CREATE OR REPLACE PIPE",
                lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            )
        ]
    )
    strategy.set_session(sess)
    result = strategy.execute_ingestion(
        config.source,
        "RAW_DATA.PUBLIC.TABLE2",
    )
    assert result.status == "failed"
    assert "fail" in (result.error or "")


def test_strategies_validate_false_when_missing_fields():
    """Test strategies validate false when missing fields."""
    # Start with valid configs, then mutate to simulate missing fields without raising at __post_init__
    cfg1 = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/data", file_format="PARQUET"
        ),
        target_table="T1",
        target_database="DB",
        target_schema="SCH",
        warehouse="WH",
        method=IngestionMethod.COPY_INTO,
    )
    strat1 = CopyIntoStrategy(cfg1)
    strat1.config.target_table = ""  # simulate missing
    assert strat1.validate() is False

    cfg2 = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage", file_format="PARQUET"
        ),
        target_table="T2",
        target_database="DB",
        target_schema="SCH",
        warehouse="WH",
        method=IngestionMethod.SNOWPIPE,
    )
    strat2 = SnowpipeStrategy(cfg2)
    strat2.config.target_table = ""  # simulate missing
    assert strat2.validate() is False
