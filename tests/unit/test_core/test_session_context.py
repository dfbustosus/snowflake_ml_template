"""Unit tests for session context."""

import pytest

from snowflake_ml_template.core.session.context import SessionContext


def test_init_schema_without_database_raises():
    """Raise when schema is provided without database."""
    with pytest.raises(ValueError):
        SessionContext(schema="FEATURES")


def test_repr_variants():
    """Test repr variants."""
    assert "SessionContext()" in repr(SessionContext())
    assert "warehouse=WH" in repr(SessionContext(warehouse="WH"))
    assert "database=DB" in repr(SessionContext(database="DB"))
    assert "schema=SC" in repr(SessionContext(database="DB", schema="SC"))


def test_enter_exit_switch_and_restore(monkeypatch: pytest.MonkeyPatch):
    """Test enter exit switch and restore."""
    calls = []

    # Previous values returned on __enter__
    monkeypatch.setattr(
        "snowflake_ml_template.core.session.context.SessionManager.get_current_warehouse",
        lambda: "PREV_WH",
    )
    monkeypatch.setattr(
        "snowflake_ml_template.core.session.context.SessionManager.get_current_database",
        lambda: "PREV_DB",
    )
    monkeypatch.setattr(
        "snowflake_ml_template.core.session.context.SessionManager.get_current_schema",
        lambda: "PREV_SC",
    )

    def switch_wh(w):
        calls.append(("switch_wh", w))

    def switch_db(d, s):
        calls.append(("switch_db", d, s))

    monkeypatch.setattr(
        "snowflake_ml_template.core.session.context.SessionManager.switch_warehouse",
        switch_wh,
    )
    monkeypatch.setattr(
        "snowflake_ml_template.core.session.context.SessionManager.switch_database",
        switch_db,
    )

    with SessionContext(warehouse="NEW_WH", database="NEW_DB", schema="NEW_SC"):
        pass

    # Verify order: apply new, then restore previous in __exit__
    assert ("switch_wh", "NEW_WH") in calls
    assert ("switch_db", "NEW_DB", "NEW_SC") in calls
    assert ("switch_db", "PREV_DB", "PREV_SC") in calls
    assert ("switch_wh", "PREV_WH") in calls


def test_exit_does_not_suppress_exception(monkeypatch: pytest.MonkeyPatch):
    """Test exit does not suppress exception."""
    # minimal stubs
    monkeypatch.setattr(
        "snowflake_ml_template.core.session.context.SessionManager.get_current_warehouse",
        lambda: None,
    )
    monkeypatch.setattr(
        "snowflake_ml_template.core.session.context.SessionManager.get_current_database",
        lambda: None,
    )
    monkeypatch.setattr(
        "snowflake_ml_template.core.session.context.SessionManager.get_current_schema",
        lambda: None,
    )

    with pytest.raises(RuntimeError):
        with SessionContext():
            raise RuntimeError("boom")
