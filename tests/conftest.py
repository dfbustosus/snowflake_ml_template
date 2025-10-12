"""Pytest helpers for unit tests.

This module ensures the project's `src` directory is on Python's import
path so test modules can import `snowflake_ml_template` directly during
CI and local runs.
"""
# Automatically add project src directory to sys.path for test imports
import os
import sys

# Determine project root (tests parent folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
