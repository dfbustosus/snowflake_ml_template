"""Infrastructure as Code for Snowflake MLOps.

This module provides infrastructure provisioning and management capabilities
for Snowflake resources. It implements the "infrastructure as code" principle,
where all infrastructure is defined in version-controlled code.

Submodules:
    provisioning: Create and manage Snowflake resources
    migrations: Schema migration management
    governance: Tags, policies, and security
"""

from snowflake_ml_template.infrastructure.provisioning import (
    DatabaseProvisioner,
    RoleProvisioner,
    SchemaProvisioner,
    StageProvisioner,
    WarehouseProvisioner,
)

__all__ = [
    "DatabaseProvisioner",
    "SchemaProvisioner",
    "RoleProvisioner",
    "WarehouseProvisioner",
    "StageProvisioner",
]
