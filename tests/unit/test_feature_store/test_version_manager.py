"""Tests for feature version manager using Snowflake metadata views."""

import pytest

from snowflake_ml_template.feature_store.versioning.manager import FeatureVersionManager


class SessionStub:
    """Stub session capturing SQL queries."""

    def __init__(self, responses):
        """Stub init."""
        self.responses = responses
        self.queries = []

    def sql(self, query):
        """Stub sql."""
        self.queries.append(query)

        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def collect(self):
                return self._rows

            def bind(self, *args, **kwargs):
                return self

        return _Result(self.responses.pop(0) if self.responses else [])


def test_list_versions_queries_catalog():
    """Ensure list_versions reads SNOWFLAKE metadata view."""
    responses = [[{"VERSION": "1.0.0"}, {"VERSION": "1.1.0"}]]
    session = SessionStub(responses)
    manager = FeatureVersionManager(session, "TEST_DB", "FEATURES")

    versions = manager.list_versions("CUSTOMER_FEATURES")

    assert versions == ["1.0.0", "1.1.0"]
    assert "SNOWFLAKE.ML.FEATURE_STORE.FEATURE_VIEW_VERSIONS" in session.queries[0]


def test_version_exists_uses_catalog_view():
    """version_exists should query metadata view."""
    responses = [[{"VERSION": "1.0.0"}]]
    session = SessionStub(responses)
    manager = FeatureVersionManager(session, "TEST_DB", "FEATURES")

    assert manager.version_exists("CUSTOMER_FEATURES", "1.0.0")
    assert "FEATURE_VIEW_VERSIONS" in session.queries[0]


def test_create_version_raises():
    """create_version is delegated to Snowflake APIs and should raise."""
    session = SessionStub([])
    manager = FeatureVersionManager(session, "TEST_DB", "FEATURES")

    with pytest.raises(Exception):
        manager.create_version(  # type: ignore[arg-type]
            feature_view="fv", version="1.0.0"
        )


def test_compare_versions_returns_diff():
    """compare_versions should return differences based on metadata view."""
    responses = [
        [
            {
                "VERSION": "1.0.0",
                "SCHEMA_HASH": "AAA",
                "FEATURE_NAMES": ["F1", "F2"],
            },
            {
                "VERSION": "1.1.0",
                "SCHEMA_HASH": "BBB",
                "FEATURE_NAMES": ["F1", "F3"],
            },
        ]
    ]
    session = SessionStub(responses)
    manager = FeatureVersionManager(session, "TEST_DB", "FEATURES")

    diff = manager.compare_versions("CUSTOMER_FEATURES", "1.0.0", "1.1.0")

    assert diff["schema_changed"] is True
    assert diff["features_added"] == ["F3"]
    assert diff["features_removed"] == ["F2"]
