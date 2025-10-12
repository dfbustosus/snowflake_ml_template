"""Lightweight wrappers and helpers for working with the Snowflake Feature Store.

These functions are intentionally conservative: they build objects or return
callable factories that require an actual Snowflake session to execute.
"""

from .api import FeatureStoreHelper

__all__ = ["FeatureStoreHelper"]
