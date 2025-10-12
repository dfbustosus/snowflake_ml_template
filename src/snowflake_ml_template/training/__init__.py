"""Training helpers: stored procedure templates, artifact staging helpers."""

from .sproc import render_training_sproc, stage_artifact_path

__all__ = ["render_training_sproc", "stage_artifact_path"]
