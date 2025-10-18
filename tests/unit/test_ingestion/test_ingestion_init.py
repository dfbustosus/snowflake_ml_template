"""Unit tests for ingestion init."""

from snowflake_ml_template.ingestion import __all__


def test_ingestion_public_api():
    """Test ingestion public API."""
    # Ensure the public API exports expected symbols
    assert set(__all__) == {
        "IngestionOrchestrator",
        "SnowpipeStrategy",
        "CopyIntoStrategy",
    }
