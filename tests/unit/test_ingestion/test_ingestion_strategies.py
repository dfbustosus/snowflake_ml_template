"""Tests for ingestion strategies."""

from snowflake_ml_template.core.base.ingestion import (
    DataSource,
    IngestionConfig,
    IngestionMethod,
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

    result = strategy.ingest(config.source, "TABLE1")
    assert result.status == "success"
    assert result.target_table == "TABLE1"


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
    )
    strategy = SnowpipeStrategy(config)
    strategy.set_session(mock_session)

    result = strategy.ingest(config.source, "TABLE2")
    assert result.status == "success"
    assert result.target_table == "TABLE2"
