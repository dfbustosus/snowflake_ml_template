"""Core FeatureStore implementation."""

import os
from typing import Any, Dict, List, Optional, cast

from snowflake.snowpark import Session

from snowflake_ml_template.core.exceptions import FeatureStoreError
from snowflake_ml_template.feature_store.core.entity import Entity
from snowflake_ml_template.feature_store.core.feature_view import FeatureView
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from snowflake.ml.feature_store.entity import Entity as SnowflakeEntity
    from snowflake.ml.feature_store.feature_view import (
        FeatureView as SnowflakeFeatureView,
    )

    _SNOWFLAKE_FS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SnowflakeEntity = cast(Any, None)
    SnowflakeFeatureView = cast(Any, None)
    _SNOWFLAKE_FS_AVAILABLE = False


class FeatureStore:
    """Enterprise Feature Store for Snowflake ML.

    Note:
        The provided `session` **must** expose a `feature_store` attribute that
        implements the Snowflake Feature Store client interface (e.g.
        `snowflake.ml.feature_store.FeatureStore`). Use
        `FeatureStore.attach_feature_store_client()` to bind the client when
        instantiating from a bare Snowpark session.
    """

    def __init__(
        self,
        session: Session,
        database: str,
        schema: str = "FEATURES",
        *,
        governance: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the FeatureStore."""
        if session is None:
            raise ValueError("Session cannot be None")
        if not database:
            raise ValueError("Database cannot be empty")

        self.session = session
        self.database = database
        self.schema = schema
        self._entities: Dict[str, Entity] = {}
        self._feature_views: Dict[str, FeatureView] = {}
        self._feature_store_client = getattr(session, "feature_store", None)
        self._default_warehouse = (
            getattr(session, "default_warehouse", None)
            or getattr(self._feature_store_client, "default_warehouse", None)
            or os.getenv("SNOWFLAKE_WAREHOUSE")
            or "COMPUTE_WH"
        )
        self._governance_config: Dict[str, Any] = governance or {}

        self._ensure_schema_exists()
        logger.info(f"Initialized Feature Store: {self.database}.{self.schema}")

    # ------------------------------------------------------------------
    # Snowflake integration helpers
    # ------------------------------------------------------------------
    def _ensure_schema_exists(self) -> None:
        """Ensure the feature store schema exists."""
        try:
            self.session.sql(
                f"CREATE SCHEMA IF NOT EXISTS {self.database}.{self.schema}"
            ).collect()
            self.session.sql(f"USE SCHEMA {self.database}.{self.schema}").collect()
        except Exception as exc:  # pragma: no cover - exercised via stubs
            raise FeatureStoreError(
                f"Failed to create schema {self.database}.{self.schema}",
                original_error=exc,
            )

    def _get_feature_store_client(self) -> Any:
        """Return the Snowflake feature store client."""
        if self._feature_store_client is None:
            raise FeatureStoreError(
                "Session is not configured with a feature store client. "
                "Set 'session.feature_store' before using FeatureStore."
            )
        return self._feature_store_client

    def _execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL and return collected rows."""
        logger.debug("Executing SQL", extra={"sql": sql})
        result = self.session.sql(sql)
        rows = result.collect()
        return cast(List[Dict[str, Any]], rows)

    def _qualify_name(self, object_name: str) -> str:
        """Return fully qualified object name."""
        return f"{self.database}.{self.schema}.{object_name}"

    def _feature_view_table_name(self, feature_view: FeatureView) -> str:
        version_suffix = feature_view.version.replace(".", "_")
        return f"FEATURE_VIEW_{feature_view.name}_V{version_suffix}"

    @staticmethod
    def _quote_identifier(value: str) -> str:
        if not value:
            raise ValueError("Identifier cannot be empty")
        escaped = value.replace('"', '""')
        return f'"{escaped}"'

    @staticmethod
    def _quote_literal(value: Optional[str]) -> str:
        if value is None:
            return "NULL"
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    def _qualify_tag(self, tag_name: str) -> str:
        parts = [part for part in tag_name.split(".") if part]
        if not parts:
            raise FeatureStoreError("Tag name cannot be empty")
        return ".".join(self._quote_identifier(part) for part in parts)

    def _apply_feature_view_governance(self, feature_view: FeatureView) -> None:
        if not self._governance_config:
            return

        per_view = self._governance_config.get("feature_views", {})
        config = per_view.get(feature_view.name)
        if not isinstance(config, dict):
            return

        table_name = self._feature_view_table_name(feature_view)
        qualified_name = self._qualify_name(table_name)
        object_type = "DYNAMIC TABLE" if feature_view.is_snowflake_managed else "TABLE"

        tags = config.get("tags", {})
        if isinstance(tags, dict):
            for tag_name, tag_value in tags.items():
                qualified_tag = self._qualify_tag(tag_name)
                literal = self._quote_literal(tag_value)
                sql = (
                    f"ALTER {object_type} {qualified_name} "
                    f"SET TAG {qualified_tag} = {literal}"
                )
                self._execute_sql(sql)

        masking_policies = config.get("masking_policies", {})
        if isinstance(masking_policies, dict):
            for column_name, policy_name in masking_policies.items():
                column_identifier = self._quote_identifier(column_name)
                sql = (
                    f"ALTER {object_type} {qualified_name} "
                    f"MODIFY COLUMN {column_identifier} SET MASKING POLICY {policy_name}"
                )
                self._execute_sql(sql)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_entity(self, entity: Entity) -> None:
        """Register an entity in the Feature Store."""
        client = self._get_feature_store_client()

        method = None
        if hasattr(client, "create_or_replace_entity"):
            method = client.create_or_replace_entity
        elif hasattr(client, "register_entity"):
            method = client.register_entity
        else:
            raise FeatureStoreError(
                "Feature store client does not expose an entity registration API"
            )

        payload_entity: Any = entity
        if _SNOWFLAKE_FS_AVAILABLE and SnowflakeEntity is not None:
            payload_entity = SnowflakeEntity(
                name=entity.name,
                join_keys=entity.join_keys,
                desc=entity.description or "",
            )

        call_kwargs = {
            "name": entity.name,
            "join_keys": entity.join_keys,
            "description": entity.description,
            "owner": entity.owner,
            "tags": entity.tags,
        }

        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((), {k: v for k, v in call_kwargs.items() if v is not None})
        ]
        if _SNOWFLAKE_FS_AVAILABLE and SnowflakeEntity is not None:
            attempts.append(((), {"entity": payload_entity}))
            attempts.append(((payload_entity,), {}))

        success = False
        for args, kwargs in attempts:
            try:
                method(*args, **kwargs)
                success = True
                break
            except TypeError:
                continue

        if not success:
            raise FeatureStoreError(
                "Feature store client signature mismatch when registering entity"
            )

        self._entities[entity.name] = entity
        logger.info(
            "Registered entity",
            extra={"entity": entity.name, "join_keys": entity.join_keys},
        )

    def _resolve_warehouse(self, feature_view: FeatureView) -> Optional[str]:
        fv_warehouse = getattr(feature_view, "warehouse", None)
        if fv_warehouse:
            return cast(Optional[str], fv_warehouse)

        session_wh_callable = getattr(self.session, "get_current_warehouse", None)
        if callable(session_wh_callable):
            try:
                session_wh = session_wh_callable()
                if isinstance(session_wh, str) and session_wh:
                    return session_wh
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to resolve session warehouse",
                    {"error": repr(exc)},
                )

        for attr in ("warehouse", "default_warehouse"):
            value = getattr(self.session, attr, None)
            if isinstance(value, str) and value:
                return value

        client = self._feature_store_client
        if client is not None:
            for attr in ("default_warehouse", "warehouse"):
                value = getattr(client, attr, None)
                if isinstance(value, str) and value:
                    return value

        env_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        if env_warehouse:
            return env_warehouse

        return self._default_warehouse

    def register_feature_view(
        self, feature_view: FeatureView, overwrite: bool = False
    ) -> None:
        """Register a feature view in the Feature Store."""
        existing = self._feature_views.get(feature_view.name)
        if existing and not overwrite:
            raise FeatureStoreError(
                f"FeatureView {feature_view.name} already exists. "
                "Use overwrite=True to replace."
            )

        client = self._get_feature_store_client()

        query: Optional[str] = None
        if feature_view.is_snowflake_managed:
            query = self._resolve_feature_df_sql(feature_view.feature_df)
            table_name = self._feature_view_table_name(feature_view)
            qualified_name = self._qualify_name(table_name)

            warehouse_name = self._resolve_warehouse(feature_view)
            if not warehouse_name:
                raise FeatureStoreError(
                    "Managed feature views require a warehouse. Set FeatureView.warehouse or ensure the session has an active warehouse."
                )
            warehouse_identifier = (
                warehouse_name
                if isinstance(warehouse_name, str)
                else self._quote_identifier(str(warehouse_name))
            )
            if (
                isinstance(warehouse_identifier, str)
                and not warehouse_identifier.isupper()
            ):
                warehouse_identifier = self._quote_identifier(warehouse_identifier)

            if overwrite:
                # Order matters: drop dynamic table first, followed by view/table fallbacks.
                self._execute_sql(f"DROP DYNAMIC TABLE IF EXISTS {qualified_name}")
                self._execute_sql(f"DROP VIEW IF EXISTS {qualified_name}")
                self._execute_sql(f"DROP TABLE IF EXISTS {qualified_name}")
            else:
                existing_objects = self._execute_sql(
                    f"SHOW OBJECTS LIKE '{table_name}' IN SCHEMA {self.database}.{self.schema}"
                )
                for obj in existing_objects:
                    kind = obj.get("kind") or obj.get("KIND")
                    if not isinstance(kind, str):
                        continue
                    kind_upper = kind.upper()
                    if kind_upper in {"TABLE", "VIEW"}:
                        message = (
                            "Conflicting object exists for feature view "
                            f"{feature_view.name}: {self.database}.{self.schema}.{table_name}"
                        )
                        raise FeatureStoreError(
                            message
                            + ". Pass overwrite=True to replace it with a dynamic table."
                        )

            target_lag = feature_view.refresh_freq or "1 hour"
            dynamic_sql = (
                f"CREATE OR REPLACE DYNAMIC TABLE {qualified_name} "
                f"TARGET_LAG = '{target_lag}' "
                f"WAREHOUSE = {warehouse_identifier} AS {query}"
            )
            self._execute_sql(dynamic_sql)
        else:
            self._validate_external_object(feature_view)

        method = None
        if hasattr(client, "create_or_replace_feature_view"):
            method = client.create_or_replace_feature_view
        elif hasattr(client, "register_feature_view"):
            method = client.register_feature_view
        else:
            raise FeatureStoreError(
                "Feature store client does not expose a feature view registration API"
            )

        snowflake_payload = None
        if (
            _SNOWFLAKE_FS_AVAILABLE
            and SnowflakeFeatureView is not None
            and SnowflakeEntity is not None
        ):
            sf_entities = [
                SnowflakeEntity(
                    name=e.name, join_keys=e.join_keys, desc=e.description or ""
                )
                for e in feature_view.entities
            ]
            fv_warehouse = getattr(feature_view, "warehouse", None)
            session_wh_callable = getattr(self.session, "get_current_warehouse", None)
            session_wh = (
                session_wh_callable() if callable(session_wh_callable) else None
            )
            default_wh = fv_warehouse or session_wh
            try:
                snowflake_payload = SnowflakeFeatureView(
                    name=feature_view.name,
                    entities=sf_entities,
                    feature_df=feature_view.feature_df,
                    refresh_freq=feature_view.refresh_freq,
                    desc=feature_view.description or "",
                    warehouse=default_wh,
                )
            except Exception:
                snowflake_payload = None

        create_kwargs: Dict[str, Any] = {
            "name": feature_view.name,
            "entities": [entity.name for entity in feature_view.entities],
            "description": feature_view.description,
            "tags": feature_view.tags,
            "feature_names": feature_view.feature_names,
            "refresh_frequency": feature_view.refresh_freq,
        }
        if query is not None:
            create_kwargs["query"] = query

        attempts_fv: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        if snowflake_payload is not None:
            attempts_fv.append(((snowflake_payload,), {}))
            attempts_fv.append(((snowflake_payload, feature_view.version), {}))
            attempts_fv.append(
                (
                    (),
                    {
                        "feature_view": snowflake_payload,
                        "version": feature_view.version,
                    },
                )
            )

        attempts_fv.append(
            ((), {k: v for k, v in create_kwargs.items() if v is not None})
        )

        for args, kwargs in attempts_fv:
            try:
                method(*args, **kwargs)
                break
            except TypeError:
                continue
        else:
            raise FeatureStoreError(
                "Feature store client signature mismatch when registering feature view"
            )

        self._feature_views[feature_view.name] = feature_view
        self._apply_feature_view_governance(feature_view)
        logger.info(
            "Registered FeatureView",
            extra={
                "feature_view": feature_view.name,
                "managed": feature_view.is_snowflake_managed,
                "entities": [entity.name for entity in feature_view.entities],
            },
        )

    # ------------------------------------------------------------------
    # Metadata & validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def attach_feature_store_client(
        session: Session, feature_store_client: Any
    ) -> None:
        """Attach a Snowflake Feature Store client to a Snowpark session."""
        setattr(session, "feature_store", feature_store_client)

    def _resolve_feature_df_sql(self, feature_df: Any) -> str:
        """Extract SQL text from a Snowpark DataFrame-like object."""
        if hasattr(feature_df, "to_sql"):
            sql = feature_df.to_sql()
            if isinstance(sql, tuple):
                sql = sql[0]
            if isinstance(sql, str):
                return sql.strip().rstrip(";")

        plan = getattr(feature_df, "_plan", None)
        if plan is not None:
            plan_queries = getattr(plan, "queries", None)
            if isinstance(plan_queries, list) and plan_queries:
                candidate = plan_queries[-1]
                text = self._extract_sql_from_candidate(candidate)
                if text:
                    return text

        if hasattr(feature_df, "queries"):
            candidate = getattr(feature_df, "queries")
            if isinstance(candidate, list) and candidate:
                text = self._extract_sql_from_candidate(candidate[-1])
                if text:
                    return text

        raise FeatureStoreError(
            "Unable to derive SQL for FeatureView dynamic table creation. "
            "Ensure the feature DataFrame implements a 'to_sql()' method or expose plan metadata."
        )

    @staticmethod
    def _extract_sql_from_candidate(candidate: Any) -> Optional[str]:
        """Best-effort extraction of SQL text from plan metadata."""
        if isinstance(candidate, str):
            return candidate.strip().rstrip(";")
        if isinstance(candidate, dict):
            text = (
                candidate.get("query") or candidate.get("text") or candidate.get("sql")
            )
            if text:
                return str(text).strip().rstrip(";")
        for attr in ("sql", "sql_text", "text"):
            if hasattr(candidate, attr):
                value = getattr(candidate, attr)
                if value:
                    return str(value).strip().rstrip(";")
        try:
            as_str = str(candidate)
        except Exception:  # pragma: no cover - defensive
            return None
        return as_str.strip().rstrip(";") if as_str else None

    def _validate_external_object(self, feature_view: FeatureView) -> None:
        """Validate external FeatureView backing object exists."""
        table_name = self._feature_view_table_name(feature_view)
        schema_qualifier = f"{self.database}.{self.schema}"
        try:
            result = self._execute_sql(
                f"SHOW TABLES LIKE '{table_name}' IN SCHEMA {schema_qualifier}"
            )
            if not result:
                # Check dynamic tables as well in case managed elsewhere
                result = self._execute_sql(
                    f"SHOW DYNAMIC TABLES LIKE '{table_name}' IN SCHEMA {schema_qualifier}"
                )
            if not result:
                raise FeatureStoreError(
                    f"External feature view backing object not found: {schema_qualifier}.{table_name}"
                )
        except Exception as exc:
            raise FeatureStoreError(
                f"Error validating backing object for {feature_view.name}",
                original_error=exc,
            )

    # ------------------------------------------------------------------
    # Metadata queries
    # ------------------------------------------------------------------
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        cached = self._entities.get(name)
        if cached:
            return cached
        return None

    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        """Get a feature view by name."""
        cached = self._feature_views.get(name)
        if cached:
            return cached
        return None

    def list_entities(self) -> List[str]:
        """List entities registered in Snowflake Feature Store."""
        names = set(self._entities.keys())
        try:
            sql = (
                "SELECT NAME FROM SNOWFLAKE.ML.FEATURE_STORE.ENTITIES "
                f"WHERE DATABASE_NAME = '{self.database}' "
                f"AND SCHEMA_NAME = '{self.schema}'"
            )
            rows = self._execute_sql(sql)
            for row in rows:
                value = row.get("NAME")
                if isinstance(value, str):
                    names.add(value)
        except Exception:
            # Fallback to cached entries only
            logger.debug("Failed to query SNOWFLAKE.ML.FEATURE_STORE.ENTITIES")
        return sorted(names)

    def list_feature_views(self) -> List[str]:
        """List feature views registered in Snowflake Feature Store."""
        names = set(self._feature_views.keys())
        try:
            sql = (
                "SELECT NAME FROM SNOWFLAKE.ML.FEATURE_STORE.FEATURE_VIEWS "
                f"WHERE DATABASE_NAME = '{self.database}' "
                f"AND SCHEMA_NAME = '{self.schema}'"
            )
            rows = self._execute_sql(sql)
            for row in rows:
                value = row.get("NAME")
                if isinstance(value, str):
                    names.add(value)
        except Exception:
            logger.debug("Failed to query SNOWFLAKE.ML.FEATURE_STORE.FEATURE_VIEWS")
        return sorted(names)

    # ------------------------------------------------------------------
    # Dynamic table lifecycle helpers
    # ------------------------------------------------------------------
    def suspend_feature_view(self, name: str) -> None:
        """Suspend a managed feature view dynamic table."""
        feature_view = self._require_feature_view(name)
        qualified = self._qualify_name(self._feature_view_table_name(feature_view))
        self._execute_sql(f"ALTER DYNAMIC TABLE {qualified} SUSPEND")

    def resume_feature_view(self, name: str) -> None:
        """Resume a managed feature view dynamic table."""
        feature_view = self._require_feature_view(name)
        qualified = self._qualify_name(self._feature_view_table_name(feature_view))
        self._execute_sql(f"ALTER DYNAMIC TABLE {qualified} RESUME")

    def refresh_feature_view(self, name: str) -> None:
        """Trigger an immediate refresh of a managed feature view dynamic table."""
        feature_view = self._require_feature_view(name)
        qualified = self._qualify_name(self._feature_view_table_name(feature_view))
        self._execute_sql(f"ALTER DYNAMIC TABLE {qualified} REFRESH")

    def get_feature_view_refresh_history(
        self, name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return recent refresh history for a managed feature view."""
        feature_view = self._require_feature_view(name)
        table_name = self._feature_view_table_name(feature_view)
        sql = (
            "SELECT * FROM INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY "
            f"WHERE TABLE_CATALOG = '{self.database}' "
            f"AND TABLE_SCHEMA = '{self.schema}' "
            f"AND TABLE_NAME = '{table_name}' "
            "ORDER BY START_TIME DESC "
            f"LIMIT {limit}"
        )
        return self._execute_sql(sql)

    def _require_feature_view(self, name: str) -> FeatureView:
        feature_view = self.get_feature_view(name)
        if not feature_view:
            raise FeatureStoreError(f"Unknown feature view: {name}")
        if feature_view.is_external:
            raise FeatureStoreError(
                f"Feature view {name} is external and has no dynamic table lifecycle"
            )
        return feature_view
