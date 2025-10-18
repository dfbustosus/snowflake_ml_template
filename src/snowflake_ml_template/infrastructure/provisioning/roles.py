"""Role provisioning for Snowflake MLOps RBAC.

This module provides functionality to create and manage Snowflake roles
following the persona-based RBAC pattern for ML workloads.

Classes:
    RoleProvisioner: Create and manage RBAC roles
"""

from typing import Dict, Iterable, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.base.tracking import ExecutionEventTracker
from snowflake_ml_template.infrastructure.provisioning.base import BaseProvisioner


class RoleProvisioner(BaseProvisioner):
    """Provision and manage Snowflake RBAC roles.

    This class handles the creation and management of Snowflake roles
    following the Golden Migration Plan's persona-based RBAC pattern:
    - ML_DATA_SCIENTIST: Data scientists (read access to prod, full access to dev)
    - ML_ENGINEER: ML engineers (operational access)
    - ML_PROD_ADMIN: Production administrators (max 3 people)
    - ML_INFERENCE_SERVICE_ROLE: Service role for inference

    Attributes:
        session: Snowflake session
        logger: Structured logger

    Example:
        >>> provisioner = RoleProvisioner(session)
        >>>
        >>> # Create all MLOps roles
        >>> provisioner.create_mlops_roles()
        >>>
        >>> # Grant role to user
        >>> provisioner.grant_role_to_user(
        ...     role="ML_DATA_SCIENTIST",
        ...     user="john.doe@company.com"
        ... )
    """

    # Standard MLOps roles
    MLOPS_ROLES = {
        "ML_DATA_SCIENTIST": "Data scientists with read access to production",
        "ML_ENGINEER": "ML engineers with operational access",
        "ML_PROD_ADMIN": "Production administrators (limited to 3 people)",
        "ML_INFERENCE_SERVICE_ROLE": "Service role for model inference",
    }

    def __init__(
        self,
        session: Session,
        tracker: Optional[ExecutionEventTracker] = None,
    ) -> None:
        """Initialize the RoleProvisioner."""
        super().__init__(session=session, tracker=tracker)

    def create_mlops_roles(self) -> Dict[str, bool]:
        """Create all standard MLOps roles.

        This method creates the four standard roles for MLOps workloads.

        Returns:
            Dictionary mapping role names to creation status

        Example:
            >>> results = provisioner.create_mlops_roles()
            >>> # {'ML_DATA_SCIENTIST': True, 'ML_ENGINEER': True, ...}
        """
        self.logger.info("Creating MLOps roles")

        results = {}

        for role, comment in self.MLOPS_ROLES.items():
            results[role] = self.create_role(name=role, comment=comment)

        self.logger.info("MLOps roles created successfully", extra={"results": results})

        return results

    def create_role(self, name: str, comment: Optional[str] = None) -> bool:
        """Create a role if it doesn't exist.

        This method is idempotent - it will not fail if the role
        already exists.

        Args:
            name: Role name
            comment: Optional comment describing the role

        Returns:
            True if role was created or already exists

        Raises:
            ConfigurationError: If role creation fails

        Example:
            >>> provisioner.create_role(
            ...     name="CUSTOM_ROLE",
            ...     comment="Custom role for specific use case"
            ... )
        """
        if not name:
            raise ValueError("Role name cannot be empty")

        self.logger.info("Creating role", extra={"role": name})

        role_identifier = self.quote_identifier(name)
        sql = f"CREATE ROLE IF NOT EXISTS {role_identifier}"
        if comment:
            sql += f" COMMENT = {self.quote_literal(comment)}"

        self._execute_sql(
            sql,
            context={"role": name},
            emit_event="role_created",
        )
        return True

    def grant_role_to_user(self, role: str, user: str) -> bool:
        """Grant a role to a user.

        Args:
            role: Role name
            user: User name or email

        Returns:
            True if grant was successful

        Raises:
            ConfigurationError: If grant fails

        Example:
            >>> provisioner.grant_role_to_user(
            ...     role="ML_DATA_SCIENTIST",
            ...     user="john.doe@company.com"
            ... )
        """
        if not role or not user:
            raise ValueError("Role and user cannot be empty")

        self.logger.info("Granting role to user", extra={"role": role, "user": user})

        sql = (
            f"GRANT ROLE {self.quote_identifier(role)} "
            f"TO USER {self.quote_identifier(user)}"
        )
        self._execute_sql(
            sql,
            context={"role": role, "user": user},
            emit_event="role_granted_to_user",
        )
        return True

    def grant_role_to_role(self, child_role: str, parent_role: str) -> bool:
        """Grant a role to another role (role hierarchy).

        Args:
            child_role: Child role name
            parent_role: Parent role name

        Returns:
            True if grant was successful

        Raises:
            ConfigurationError: If grant fails
        """
        if not child_role or not parent_role:
            raise ValueError("Child and parent roles cannot be empty")

        self.logger.info(
            "Granting role to role",
            extra={"child_role": child_role, "parent_role": parent_role},
        )

        sql = (
            f"GRANT ROLE {self.quote_identifier(child_role)} "
            f"TO ROLE {self.quote_identifier(parent_role)}"
        )
        self._execute_sql(
            sql,
            context={"child_role": child_role, "parent_role": parent_role},
            emit_event="role_granted_to_role",
        )
        return True

    def grant_database_privileges(
        self, role: str, database: str, privileges: List[str]
    ) -> bool:
        """Grant database privileges to a role.

        Args:
            role: Role name
            database: Database name
            privileges: List of privileges (e.g., ['USAGE', 'CREATE SCHEMA'])

        Returns:
            True if grant was successful

        Raises:
            ConfigurationError: If grant fails

        Example:
            >>> provisioner.grant_database_privileges(
            ...     role="ML_DATA_SCIENTIST",
            ...     database="ML_DEV_DB",
            ...     privileges=["USAGE", "CREATE SCHEMA"]
            ... )
        """
        if not role or not database or not privileges:
            raise ValueError("Role, database, and privileges cannot be empty")

        self.logger.info(
            "Granting database privileges",
            extra={"role": role, "database": database, "privileges": privileges},
        )

        database_identifier = self.quote_identifier(database)
        role_identifier = self.quote_identifier(role)
        for privilege in privileges:
            sql = (
                f"GRANT {privilege} ON DATABASE {database_identifier} "
                f"TO ROLE {role_identifier}"
            )
            self._execute_sql(
                sql,
                context={"role": role, "database": database, "privilege": privilege},
                emit_event="database_privilege_granted",
            )
        return True

    def grant_schema_privileges(
        self, role: str, database: str, schema: str, privileges: List[str]
    ) -> bool:
        """Grant schema privileges to a role.

        Args:
            role: Role name
            database: Database name
            schema: Schema name
            privileges: List of privileges (e.g., ['USAGE', 'CREATE TABLE'])

        Returns:
            True if grant was successful

        Raises:
            ConfigurationError: If grant fails
        """
        if not role or not database or not schema or not privileges:
            raise ValueError("Role, database, schema, and privileges cannot be empty")

        full_schema = f"{database}.{schema}"
        self.logger.info(
            "Granting schema privileges",
            extra={"role": role, "schema": full_schema, "privileges": privileges},
        )

        role_identifier = self.quote_identifier(role)
        schema_identifier = self.format_qualified_identifier(database, schema)
        for privilege in privileges:
            sql = (
                f"GRANT {privilege} ON SCHEMA {schema_identifier} "
                f"TO ROLE {role_identifier}"
            )
            self._execute_sql(
                sql,
                context={"role": role, "schema": full_schema, "privilege": privilege},
                emit_event="schema_privilege_granted",
            )
        return True

    def grant_warehouse_privileges(
        self, role: str, warehouse: str, privileges: List[str]
    ) -> bool:
        """Grant warehouse privileges to a role.

        Args:
            role: Role name
            warehouse: Warehouse name
            privileges: List of privileges (e.g., ['USAGE', 'OPERATE'])

        Returns:
            True if grant was successful

        Raises:
            ConfigurationError: If grant fails
        """
        if not role or not warehouse or not privileges:
            raise ValueError("Role, warehouse, and privileges cannot be empty")

        self.logger.info(
            "Granting warehouse privileges",
            extra={"role": role, "warehouse": warehouse, "privileges": privileges},
        )

        role_identifier = self.quote_identifier(role)
        warehouse_identifier = self.quote_identifier(warehouse)
        for privilege in privileges:
            sql = (
                f"GRANT {privilege} ON WAREHOUSE {warehouse_identifier} "
                f"TO ROLE {role_identifier}"
            )
            self._execute_sql(
                sql,
                context={"role": role, "warehouse": warehouse, "privilege": privilege},
                emit_event="warehouse_privilege_granted",
            )
        return True

    def grant_future_schema_privileges(
        self,
        role: str,
        database: str,
        privileges: Iterable[str],
    ) -> bool:
        """Grant future schema privileges to a role.

        Args:
            role: Role name
            database: Database name
            privileges: List of privileges (e.g., ['USAGE', 'CREATE SCHEMA'])

        Returns:
            True if grant was successful

        Raises:
            ConfigurationError: If grant fails
        """
        if not role or not database:
            raise ValueError("Role and database cannot be empty")
        database_identifier = self.quote_identifier(database)
        role_identifier = self.quote_identifier(role)
        for privilege in privileges:
            sql = (
                f"GRANT {privilege} ON FUTURE SCHEMAS IN DATABASE {database_identifier} "
                f"TO ROLE {role_identifier}"
            )
            self._execute_sql(
                sql,
                context={"role": role, "database": database, "privilege": privilege},
                emit_event="future_schema_privilege_granted",
            )
        return True

    def revoke_role_from_user(self, role: str, user: str) -> bool:
        """Revoke a role from a user.

        Args:
            role: Role name
            user: User name or email

        Returns:
            True if revoke was successful

        Raises:
            ConfigurationError: If revoke fails
        """
        if not role or not user:
            raise ValueError("Role and user cannot be empty")
        sql = (
            f"REVOKE ROLE {self.quote_identifier(role)} "
            f"FROM USER {self.quote_identifier(user)}"
        )
        self._execute_sql(
            sql,
            context={"role": role, "user": user},
            emit_event="role_revoked_from_user",
        )
        return True

    def revoke_role_from_role(self, child_role: str, parent_role: str) -> bool:
        """Revoke a role from another role (role hierarchy).

        Args:
            child_role: Child role name
            parent_role: Parent role name

        Returns:
            True if revoke was successful

        Raises:
            ConfigurationError: If revoke fails
        """
        if not child_role or not parent_role:
            raise ValueError("Child and parent roles cannot be empty")
        sql = (
            f"REVOKE ROLE {self.quote_identifier(child_role)} "
            f"FROM ROLE {self.quote_identifier(parent_role)}"
        )
        self._execute_sql(
            sql,
            context={"child_role": child_role, "parent_role": parent_role},
            emit_event="role_revoked_from_role",
        )
        return True

    def role_exists(self, name: str) -> bool:
        """Check if a role exists.

        Args:
            name: Role name

        Returns:
            True if role exists, False otherwise
        """
        try:
            result = self.session.sql(
                f"SHOW ROLES LIKE {self.quote_literal(name)}"
            ).collect()
            return len(result) > 0
        except Exception:
            return False

    def list_roles(self) -> List[str]:
        """List all roles.

        Returns:
            List of role names
        """
        try:
            result = self.session.sql("SHOW ROLES").collect()
            return [row["name"] for row in result]
        except Exception as e:
            self.logger.error("Failed to list roles", extra={"error": str(e)})
            return []
