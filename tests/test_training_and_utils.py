"""Unit tests for training helpers and utility functions."""

from snowflake_ml_template.training.sproc import (
    render_training_sproc,
    stage_artifact_path,
)


def test_render_training_sproc_contains_proc():
    """Test that render_training_sproc generates a CREATE PROCEDURE statement."""
    sql = render_training_sproc()
    assert "CREATE OR REPLACE PROCEDURE" in sql


def test_stage_artifact_path():
    """Test that stage_artifact_path returns the correct stage path."""
    path = stage_artifact_path("my_model", "v1")
    assert "@ML_MODELS_STAGE/my_model/v1/" in path
