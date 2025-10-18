"""Tests for Feature Store Entity."""

import pytest

from snowflake_ml_template.feature_store.core.entity import Entity, SQLEntity


def test_entity_creation():
    """Test entity creation with valid parameters."""
    entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"])
    assert entity.name == "CUSTOMER"
    assert entity.join_keys == ["CUSTOMER_ID"]


def test_entity_validation_empty_name():
    """Test entity validation fails with empty name."""
    with pytest.raises(ValueError, match="Entity name cannot be empty"):
        Entity(name="", join_keys=["ID"])


def test_entity_validation_no_join_keys():
    """Test entity validation fails with no join keys."""
    with pytest.raises(ValueError, match="at least one join key"):
        Entity(name="CUSTOMER", join_keys=[])


def test_entity_validation_duplicate_join_keys():
    """Test entity validation fails with duplicate join keys."""
    with pytest.raises(ValueError, match="must be unique"):
        Entity(name="CUSTOMER", join_keys=["ID", "ID"])


def test_entity_get_join_key_columns():
    """Test getting join key columns."""
    entity = Entity(name="CUSTOMER", join_keys=["ID1", "ID2"])
    keys = entity.get_join_key_columns()
    assert keys == ["ID1", "ID2"]
    # Verify it returns a copy
    keys.append("ID3")
    assert entity.join_keys == ["ID1", "ID2"]


def test_sql_entity_creation():
    """Test SQL entity creation."""
    entity = SQLEntity(
        name="CUSTOMER",
        join_keys=["CUSTOMER_ID"],
        sql_definition="SELECT * FROM CUSTOMERS",
    )
    assert entity.name == "CUSTOMER"
    assert entity.sql_definition == "SELECT * FROM CUSTOMERS"
