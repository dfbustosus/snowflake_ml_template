"""Tests for ingestion strategies."""

from unittest.mock import Mock

from snowflake_ml_template.core.base.ingestion import (
    CopyIntoOptions,
    DataSource,
    IngestionConfig,
    IngestionMethod,
    SnowpipeLoadMethod,
    SnowpipeOptions,
    SourceType,
)
from snowflake_ml_template.ingestion.strategies.copy_into import CopyIntoStrategy
from snowflake_ml_template.ingestion.strategies.snowpipe import SnowpipeStrategy


def test_copy_into_strategy_initialization():
    """Test COPY INTO strategy initialization."""
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
    assert strategy.config.target_table == "TABLE1"


def test_copy_into_strategy_validation():
    """Test COPY INTO strategy validation."""
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
    assert strategy.validate() is True


def test_copy_into_strategy_get_target_table():
    """Test getting target table name."""
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
    assert strategy.get_target_table_name() == "RAW_DATA.PUBLIC.TABLE1"


def test_snowpipe_strategy_initialization():
    """Test Snowpipe strategy initialization."""
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
    assert strategy.config.target_table == "TABLE2"


def test_snowpipe_strategy_validation():
    """Test Snowpipe strategy validation."""
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
    assert strategy.validate() is True


def test_copy_into_strategy_ingest(mock_session):
    """Test COPY INTO ingestion."""
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
    strategy.set_session(mock_session)
    warehouse_call = Mock()
    warehouse_call.collect.return_value = []
    copy_call = Mock()
    copy_call.collect.return_value = [{"rows_loaded": 3}]
    mock_session.sql.side_effect = [warehouse_call, copy_call]

    result = strategy.execute_ingestion(config.source, strategy.get_target_table_name())
    assert result.status == "success"
    assert result.rows_loaded == 3
    assert result.target_table == "RAW_DATA.PUBLIC.TABLE1"


def test_copy_into_strategy_ingest_with_options(mock_session):
    """Test COPY INTO ingestion with extended options."""
    config = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/data", file_format="PARQUET"
        ),
        target_table="TABLE1",
        target_database="RAW_DATA",
        target_schema="PUBLIC",
        warehouse="TEST_WH",
        method=IngestionMethod.COPY_INTO,
        copy_options=CopyIntoOptions(
            files=["file1.csv"],
            force=True,
            parallel=8,
            enforce_length=True,
            null_if=["NULL"],
        ),
    )
    strategy = CopyIntoStrategy(config)
    strategy.set_session(mock_session)

    warehouse_call = Mock()
    warehouse_call.collect.return_value = []
    copy_call = Mock()
    copy_call.collect.return_value = [{"rows_loaded": 5}]
    mock_session.sql.side_effect = [warehouse_call, copy_call]

    result = strategy.execute_ingestion(config.source, strategy.get_target_table_name())

    assert result.status == "success"
    assert result.target_table == "RAW_DATA.PUBLIC.TABLE1"
    sql_statement = mock_session.sql.call_args_list[1].args[0]
    assert "FILES = ('file1.csv')" in sql_statement
    assert "ENFORCE_LENGTH = TRUE" in sql_statement
    assert result.metadata["copy_options"]["parallel"] == 8


def test_copy_into_strategy_includes_copy_history(mock_session):
    """COPY INTO ingestion should capture copy history metadata when available."""
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
    strategy.set_session(mock_session)

    warehouse_call = Mock()
    warehouse_call.collect.return_value = []
    copy_call = Mock()
    copy_call.collect.return_value = [{"rows_loaded": 2}]
    history_call = Mock()
    history_call.collect.return_value = [{"FILE_NAME": "file1.csv"}]
    mock_session.sql.side_effect = [warehouse_call, copy_call, history_call]

    result = strategy.execute_ingestion(
        config.source,
        strategy.get_target_table_name(),
        history_window_minutes=30,
    )

    assert result.metadata["copy_history"][0]["FILE_NAME"] == "file1.csv"


def test_snowpipe_strategy_ingest(mock_session):
    """Test Snowpipe ingestion."""
    config = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/stream", file_format="JSON"
        ),
        target_table="TABLE2",
        target_database="RAW_DATA",
        target_schema="PUBLIC",
        warehouse="TEST_WH",
        method=IngestionMethod.SNOWPIPE,
        snowpipe_options=SnowpipeOptions(auto_ingest=True, integration="PIPE_INT"),
    )
    strategy = SnowpipeStrategy(config)
    strategy.set_session(mock_session)
    create_call = Mock()
    create_call.collect.return_value = []
    status_call = Mock()
    status_call.collect.return_value = [{"STATUS": {"state": "RUNNING"}}]
    mock_session.sql.side_effect = [create_call, status_call]

    result = strategy.execute_ingestion(config.source, strategy.get_target_table_name())

    assert result.status == "success"
    assert result.metadata["pipe_status"]["state"] == "RUNNING"


def test_snowpipe_auto_ingest_requires_notification():
    """Auto-ingest configuration must provide integration or notification channel."""
    config = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/stream", file_format="JSON"
        ),
        target_table="TABLE2",
        target_database="RAW_DATA",
        target_schema="PUBLIC",
        warehouse="TEST_WH",
        method=IngestionMethod.SNOWPIPE,
        snowpipe_options=SnowpipeOptions(auto_ingest=True),
    )
    strategy = SnowpipeStrategy(config)
    assert strategy.validate() is False


def test_snowpipe_rest_refresh_invokes_force_refresh(mock_session):
    """REST load method should invoke SYSTEM$PIPE_FORCE_REFRESH."""
    config = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/stream", file_format="JSON"
        ),
        target_table="TABLE2",
        target_database="RAW_DATA",
        target_schema="PUBLIC",
        warehouse="TEST_WH",
        method=IngestionMethod.SNOWPIPE,
        snowpipe_options=SnowpipeOptions(load_method=SnowpipeLoadMethod.REST_API),
    )
    strategy = SnowpipeStrategy(config)
    strategy.set_session(mock_session)

    create_call = Mock()
    create_call.collect.return_value = []
    force_call = Mock()
    force_call.collect.return_value = []
    status_call = Mock()
    status_call.collect.return_value = []
    mock_session.sql.side_effect = [create_call, force_call, status_call]

    result = strategy.execute_ingestion(
        config.source,
        strategy.get_target_table_name(),
        files=["file1.csv"],
    )

    assert result.metadata["rest_refresh"]["files"] == ["file1.csv"]
    assert "SYSTEM$PIPE_FORCE_REFRESH" in mock_session.sql.call_args_list[1].args[0]


def test_snowpipe_streaming_uses_client(mock_session):
    """Streaming load method should delegate to streaming client."""
    config = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/stream", file_format="JSON"
        ),
        target_table="TABLE2",
        target_database="RAW_DATA",
        target_schema="PUBLIC",
        warehouse="TEST_WH",
        method=IngestionMethod.SNOWPIPE,
        snowpipe_options=SnowpipeOptions(load_method=SnowpipeLoadMethod.STREAMING),
    )
    strategy = SnowpipeStrategy(config)
    strategy.set_session(mock_session)

    create_call = Mock()
    create_call.collect.return_value = []
    status_call = Mock()
    status_call.collect.return_value = []
    mock_session.sql.side_effect = [create_call, status_call]

    class StreamingStub:
        def __init__(self) -> None:
            self.calls = []

        def insert_rows(self, target: str, rows: list[dict]) -> str:
            self.calls.append((target, rows))
            return "batch-1"

    client = StreamingStub()

    result = strategy.execute_ingestion(
        config.source,
        strategy.get_target_table_name(),
        streaming_client=client,
        rows=[{"id": 1}],
    )

    assert result.metadata["streaming"]["batch_id"] == "batch-1"
    assert client.calls[0][0] == strategy.get_target_table_name()


def test_snowpipe_streaming_uses_default_client(mock_session):
    """Streaming load method should create a SnowpipeStreamingClient when missing."""
    config = IngestionConfig(
        source=DataSource(
            source_type=SourceType.S3, location="@stage/stream", file_format="JSON"
        ),
        target_table="TABLE2",
        target_database="RAW_DATA",
        target_schema="PUBLIC",
        warehouse="TEST_WH",
        method=IngestionMethod.SNOWPIPE,
        snowpipe_options=SnowpipeOptions(
            load_method=SnowpipeLoadMethod.STREAMING,
            warehouse="INGEST_WH",
            role="INGEST_ROLE",
        ),
    )
    strategy = SnowpipeStrategy(config)
    strategy.set_session(mock_session)

    create_call = Mock()
    create_call.collect.return_value = []
    status_call = Mock()
    status_call.collect.return_value = []
    use_role_call = Mock()
    use_role_call.collect.return_value = []
    use_wh_call = Mock()
    use_wh_call.collect.return_value = []
    mock_session.sql.side_effect = [
        create_call,
        status_call,
        use_role_call,
        use_wh_call,
    ]
    mock_session.create_dataframe = Mock()
    dataframe = Mock()
    dataframe.write.mode.return_value = dataframe.write
    dataframe.write.save_as_table.return_value = None
    mock_session.create_dataframe.return_value = dataframe

    result = strategy.execute_ingestion(
        config.source,
        strategy.get_target_table_name(),
        rows=[{"id": 1, "value": "abc"}],
    )

    mock_session.create_dataframe.assert_called_once()
    assert result.metadata["streaming"]["row_count"] == 1
