"""Tests for orchestration components."""

from snowflake_ml_template.orchestration.streams import StreamProcessor
from snowflake_ml_template.orchestration.tasks import TaskOrchestrator


def test_task_orchestrator_initialization(mock_session):
    """Test task orchestrator initialization."""
    orchestrator = TaskOrchestrator(mock_session, "TEST_DB", "PIPELINES")
    assert orchestrator.session == mock_session
    assert orchestrator.database == "TEST_DB"


def test_task_orchestrator_create_task(mock_session):
    """Test task orchestrator creates task."""
    orchestrator = TaskOrchestrator(mock_session, "TEST_DB", "PIPELINES")
    result = orchestrator.create_task(
        name="test_task",
        sql="SELECT 1",
        schedule="USING CRON 0 0 * * * UTC",
        warehouse="TEST_WH",
    )
    assert result is True


def test_task_orchestrator_resume_task(mock_session):
    """Test task orchestrator resumes task."""
    orchestrator = TaskOrchestrator(mock_session, "TEST_DB", "PIPELINES")
    result = orchestrator.resume_task("test_task")
    assert result is True


def test_stream_processor_initialization(mock_session):
    """Test stream processor initialization."""
    processor = StreamProcessor(mock_session, "TEST_DB", "RAW_DATA")
    assert processor.session == mock_session
    assert processor.database == "TEST_DB"


def test_stream_processor_create_stream(mock_session):
    """Test stream processor creates stream."""
    processor = StreamProcessor(mock_session, "TEST_DB", "RAW_DATA")
    result = processor.create_stream("test_stream", "test_table")
    assert result is True


def test_stream_processor_has_data(mock_session):
    """Test stream processor checks for data."""
    mock_session.sql.return_value.collect.return_value = [[True]]
    processor = StreamProcessor(mock_session, "TEST_DB", "RAW_DATA")
    result = processor.has_data("test_stream")
    assert result is True
