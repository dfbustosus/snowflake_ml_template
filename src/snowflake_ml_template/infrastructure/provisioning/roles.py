"""Role provisioning for Snowflake MLOps RBAC.

This module provides functionality to create and manage Snowflake roles
following the persona-based RBAC pattern for ML workloads.

Classes:
    RoleProvisioner: Create and manage RBAC roles
"""

from typing import Dict, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import ConfigurationError
from snowflake_ml_template.utils.logging import get_logger


class RoleProvisioner:
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

    def __init__(self, session: Session) -> None:
        """Initialize the role provisioner.

        Args:
            session: Active Snowflake session

        Raises:
            ValueError: If session is None
        """
        if session is None:
            raise ValueError("Session cannot be None")

        self.session = session
        self.logger = get_logger(__name__)

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

        self.logger.info(f"Creating role: {name}")

        try:
            # Create role
            sql = f"CREATE ROLE IF NOT EXISTS {name}"
            if comment:
                sql += f" COMMENT = '{comment}'"

            self.session.sql(sql).collect()

            self.logger.info(f"Role created successfully: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create role: {name}", extra={"error": str(e)})
            raise ConfigurationError(
                f"Failed to create role: {name}",
                context={"role": name},
                original_error=e,
            )

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

        self.logger.info(f"Granting role {role} to user {user}")

        try:
            sql = f"GRANT ROLE {role} TO USER {user}"
            self.session.sql(sql).collect()

            self.logger.info(
                "Role granted successfully", extra={"role": role, "user": user}
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to grant role {role} to user {user}", extra={"error": str(e)}
            )
            raise ConfigurationError(
                f"Failed to grant role {role} to user {user}",
                context={"role": role, "user": user},
                original_error=e,
            )

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

        self.logger.info(f"Granting role {child_role} to role {parent_role}")

        try:
            sql = f"GRANT ROLE {child_role} TO ROLE {parent_role}"
            self.session.sql(sql).collect()

            self.logger.info(
                "Role granted successfully",
                extra={"child_role": child_role, "parent_role": parent_role},
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to grant role {child_role} to role {parent_role}",
                extra={"error": str(e)},
            )
            raise ConfigurationError(
                f"Failed to grant role {child_role} to role {parent_role}",
                context={"child_role": child_role, "parent_role": parent_role},
                original_error=e,
            )

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
            f"Granting database privileges to role {role}",
            extra={"database": database, "privileges": privileges},
        )

        try:
            for privilege in privileges:
                sql = f"GRANT {privilege} ON DATABASE {database} TO ROLE {role}"
                self.session.sql(sql).collect()

            self.logger.info(
                "Database privileges granted successfully",
                extra={"role": role, "database": database, "privileges": privileges},
            )
            return True

        except Exception as e:
            self.logger.error(
                "Failed to grant database privileges",
                extra={"role": role, "database": database, "error": str(e)},
            )
            raise ConfigurationError(
                f"Failed to grant database privileges to role {role}",
                context={"role": role, "database": database, "privileges": privileges},
                original_error=e,
            )

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
            f"Granting schema privileges to role {role}",
            extra={"schema": full_schema, "privileges": privileges},
        )

        try:
            for privilege in privileges:
                sql = f"GRANT {privilege} ON SCHEMA {full_schema} TO ROLE {role}"
                self.session.sql(sql).collect()

            self.logger.info(
                "Schema privileges granted successfully",
                extra={"role": role, "schema": full_schema, "privileges": privileges},
            )
            return True

        except Exception as e:
            self.logger.error(
                "Failed to grant schema privileges",
                extra={"role": role, "schema": full_schema, "error": str(e)},
            )
            raise ConfigurationError(
                f"Failed to grant schema privileges to role {role}",
                context={"role": role, "schema": full_schema, "privileges": privileges},
                original_error=e,
            )

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
            f"Granting warehouse privileges to role {role}",
            extra={"warehouse": warehouse, "privileges": privileges},
        )

        try:
            for privilege in privileges:
                sql = f"GRANT {privilege} ON WAREHOUSE {warehouse} TO ROLE {role}"
                self.session.sql(sql).collect()

            self.logger.info(
                "Warehouse privileges granted successfully",
                extra={"role": role, "warehouse": warehouse, "privileges": privileges},
            )
            return True

        except Exception as e:
            self.logger.error(
                "Failed to grant warehouse privileges",
                extra={"role": role, "warehouse": warehouse, "error": str(e)},
            )
            raise ConfigurationError(
                f"Failed to grant warehouse privileges to role {role}",
                context={
                    "role": role,
                    "warehouse": warehouse,
                    "privileges": privileges,
                },
                original_error=e,
            )

    def role_exists(self, name: str) -> bool:
        """Check if a role exists.

        Args:
            name: Role name

        Returns:
            True if role exists, False otherwise
        """
        try:
            result = self.session.sql(f"SHOW ROLES LIKE '{name}'").collect()
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
            self.logger.error(f"Failed to list roles: {e}")
            return []
