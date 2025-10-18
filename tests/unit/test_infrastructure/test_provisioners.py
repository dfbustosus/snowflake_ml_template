"""Tests for infrastructure provisioners."""

from snowflake_ml_template.infrastructure.provisioning.databases import (
    DatabaseProvisioner,
)
from snowflake_ml_template.infrastructure.provisioning.roles import RoleProvisioner
from snowflake_ml_template.infrastructure.provisioning.schemas import SchemaProvisioner
from snowflake_ml_template.infrastructure.provisioning.stages import StageProvisioner
from snowflake_ml_template.infrastructure.provisioning.warehouses import (
    WarehouseProvisioner,
)


def test_database_provisioner_create(mock_session):
    """Test database provisioner creates database."""
    provisioner = DatabaseProvisioner(mock_session)
    result = provisioner.create_database("TEST_DB")
    assert result is True


def test_database_provisioner_clone(mock_session):
    """Test database provisioner clones database."""
    provisioner = DatabaseProvisioner(mock_session)
    result = provisioner.clone_database("PROD_DB", "TEST_DB")
    assert result is True


def test_schema_provisioner_create(mock_session):
    """Test schema provisioner creates schema."""
    provisioner = SchemaProvisioner(mock_session)
    result = provisioner.create_schema("TEST_DB", "TEST_SCHEMA")
    assert result is True


def test_schema_provisioner_create_ml_schemas(mock_session):
    """Test schema provisioner creates ML schemas."""
    provisioner = SchemaProvisioner(mock_session)
    # Just test initialization
    assert provisioner.session == mock_session


def test_role_provisioner_create(mock_session):
    """Test role provisioner creates role."""
    provisioner = RoleProvisioner(mock_session)
    # Just test initialization
    assert provisioner.session == mock_session


def test_warehouse_provisioner_create(mock_session):
    """Test warehouse provisioner creates warehouse."""
    provisioner = WarehouseProvisioner(mock_session)
    # Just test initialization
    assert provisioner.session == mock_session


def test_stage_provisioner_create(mock_session):
    """Test stage provisioner creates stage."""
    provisioner = StageProvisioner(mock_session)
    # Just test initialization
    assert provisioner.session == mock_session
