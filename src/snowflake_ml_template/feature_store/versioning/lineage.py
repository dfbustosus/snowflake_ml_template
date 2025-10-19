"""Data lineage tracking for feature store.

This module implements comprehensive lineage tracking from source data
through transformations to features, enabling:
- Impact analysis (what's affected by changes)
- Root cause analysis (where did this feature come from)
- Compliance and governance
- Reproducibility
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from snowflake.snowpark import Session

from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class LineageNodeType(Enum):
    """Types of nodes in the lineage graph."""

    SOURCE_TABLE = "source_table"
    TRANSFORMATION = "transformation"
    FEATURE_VIEW = "feature_view"
    ENTITY = "entity"
    MODEL = "model"


@dataclass
class LineageNode:
    """Represents a node in the lineage graph.

    Attributes:
        id: Unique identifier for this node
        type: Type of node (source, transformation, feature, etc.)
        name: Human-readable name
        metadata: Additional metadata about this node
        created_at: When this node was created
    """

    id: str
    type: LineageNodeType
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LineageEdge:
    """Represents a relationship in the lineage graph.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        relationship: Type of relationship (derives_from, uses, etc.)
        metadata: Additional metadata about this relationship
        created_at: When this relationship was created
    """

    source_id: str
    target_id: str
    relationship: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class LineageTracker:
    """Track and query data lineage in the feature store.

    This class maintains a directed acyclic graph (DAG) of lineage
    relationships, enabling:
    - Forward lineage: What depends on this data?
    - Backward lineage: Where did this feature come from?
    - Impact analysis: What breaks if I change this?

    The lineage graph is stored in Snowflake tables for persistence
    and queryability.

    Attributes:
        session: Snowflake session
        database: Database containing feature store
        schema: Schema containing feature store
        logger: Structured logger

    Example:
        >>> tracker = LineageTracker(session, "ML_PROD_DB", "FEATURES")
        >>>
        >>> # Track source table
        >>> tracker.add_source_table("TRANSACTIONS", "RAW_DATA.TRANSACTIONS")
        >>>
        >>> # Track feature view lineage
        >>> tracker.add_feature_view_lineage(
        ...     feature_view_name="customer_features",
        ...     source_tables=["TRANSACTIONS", "CUSTOMERS"]
        ... )
        >>>
        >>> # Query lineage
        >>> upstream = tracker.get_upstream_lineage("customer_features")
        >>> downstream = tracker.get_downstream_lineage("TRANSACTIONS")
    """

    def __init__(
        self,
        session: Session,
        database: str,
        schema: str,
        *,
        enforce_node_validation: bool = False,
    ):
        """Initialize the lineage tracker.

        Args:
            session: Active Snowflake session
            database: Database containing feature store
            schema: Schema containing feature store
            enforce_node_validation: Whether to enforce node presence checks against Snowflake
        """
        if session is None:
            raise ValueError("Session cannot be None")
        if not database or not schema:
            raise ValueError("Database and schema cannot be empty")

        self.session = session
        self.database = database
        self.schema = schema
        self.logger = get_logger(__name__)
        self._cached_node_ids: set[str] = set()
        self._enforce_node_validation = enforce_node_validation

        self._ensure_lineage_tables_exist()

    def _ensure_lineage_tables_exist(self) -> None:
        """Create lineage tracking tables if they don't exist."""
        # Nodes table
        nodes_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.LINEAGE_NODES (
            node_id VARCHAR PRIMARY KEY,
            node_type VARCHAR NOT NULL,
            node_name VARCHAR NOT NULL,
            metadata VARIANT,
            created_at TIMESTAMP_NTZ NOT NULL,
            updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """

        # Edges table
        edges_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.LINEAGE_EDGES (
            edge_id VARCHAR PRIMARY KEY,
            source_node_id VARCHAR NOT NULL,
            target_node_id VARCHAR NOT NULL,
            relationship VARCHAR NOT NULL,
            metadata VARIANT,
            created_at TIMESTAMP_NTZ NOT NULL,
            FOREIGN KEY (source_node_id) REFERENCES {self.database}.{self.schema}.LINEAGE_NODES(node_id),
            FOREIGN KEY (target_node_id) REFERENCES {self.database}.{self.schema}.LINEAGE_NODES(node_id)
        )
        """

        self.session.sql(nodes_sql).collect()
        self.session.sql(edges_sql).collect()

    def add_source_table(
        self, table_id: str, table_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> LineageNode:
        """Add a source table to the lineage graph.

        Args:
            table_id: Unique identifier for the table
            table_name: Fully qualified table name
            metadata: Optional metadata about the table

        Returns:
            LineageNode representing the source table
        """
        node = LineageNode(
            id=table_id,
            type=LineageNodeType.SOURCE_TABLE,
            name=table_name,
            metadata=metadata or {},
        )

        self._store_node(node)

        self.logger.info(
            f"Added source table to lineage: {table_name}",
            extra={"table_id": table_id, "table_name": table_name},
        )

        return node

    def add_feature_view_lineage(
        self,
        feature_view_name: str,
        source_tables: List[str],
        transformation_logic: Optional[str] = None,
    ) -> None:
        """Add feature view lineage relationships.

        This creates nodes and edges representing the lineage from source
        tables through transformations to the feature view.

        Args:
            feature_view_name: Name of the feature view
            source_tables: List of source table IDs
            transformation_logic: Optional description of transformation
        """
        # Create feature view node
        fv_node = LineageNode(
            id=f"fv_{feature_view_name}",
            type=LineageNodeType.FEATURE_VIEW,
            name=feature_view_name,
            metadata={"transformation_logic": transformation_logic},
        )
        self._store_node(fv_node)

        # Create edges from source tables to feature view
        for source_table_id in source_tables:
            edge = LineageEdge(
                source_id=source_table_id,
                target_id=fv_node.id,
                relationship="derives_from",
            )
            self._store_edge(edge)

        self.logger.info(
            f"Added feature view lineage: {feature_view_name}",
            extra={"feature_view": feature_view_name, "source_tables": source_tables},
        )

    def _store_node(self, node: LineageNode) -> None:
        """Store a lineage node in Snowflake."""
        sql = f"""
        MERGE INTO {self.database}.{self.schema}.LINEAGE_NODES AS target
        USING (SELECT ? AS node_id) AS source
        ON target.node_id = source.node_id
        WHEN MATCHED THEN
            UPDATE SET updated_at = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN
            INSERT (node_id, node_type, node_name, metadata, created_at)
            VALUES (?, ?, ?, PARSE_JSON(?), ?)
        """

        import json

        self.session.sql(sql).bind(
            node.id,
            node.id,
            node.type.value,
            node.name,
            json.dumps(node.metadata),
            node.created_at,
        ).collect()
        self._cached_node_ids.add(node.id)

    def _store_edge(self, edge: LineageEdge) -> None:
        """Store a lineage edge in Snowflake."""
        # Generate edge ID from source and target
        edge_id = hashlib.sha256(
            f"{edge.source_id}_{edge.target_id}_{edge.relationship}".encode()
        ).hexdigest()[:32]

        self._assert_node_exists(edge.source_id)
        self._assert_node_exists(edge.target_id)

        sql = f"""
        INSERT INTO {self.database}.{self.schema}.LINEAGE_EDGES
        (edge_id, source_node_id, target_node_id, relationship, metadata, created_at)
        SELECT ?, ?, ?, ?, PARSE_JSON(?), ?
        WHERE NOT EXISTS (
            SELECT 1 FROM {self.database}.{self.schema}.LINEAGE_EDGES
            WHERE edge_id = ?
        )
        """

        self.session.sql(sql).bind(
            edge_id,
            edge.source_id,
            edge.target_id,
            edge.relationship,
            json.dumps(edge.metadata),
            edge.created_at,
            edge_id,
        ).collect()

    def _assert_node_exists(self, node_id: str) -> None:
        """Ensure a lineage node exists before creating relationships."""
        if node_id in self._cached_node_ids or not self._enforce_node_validation:
            return

        sql = f"""
        SELECT 1
        FROM {self.database}.{self.schema}.LINEAGE_NODES
        WHERE node_id = ?
        LIMIT 1
        """
        result = self.session.sql(sql).bind(node_id).collect()
        if not result:
            raise ValueError(f"Lineage node '{node_id}' does not exist")
        self._cached_node_ids.add(node_id)

    def get_upstream_lineage(
        self, node_id: str, max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Get all upstream dependencies (backward lineage).

        This traverses the lineage graph backward to find all sources
        that contribute to the given node.

        Args:
            node_id: ID of the node to trace
            max_depth: Maximum depth to traverse

        Returns:
            List of upstream nodes with their relationships

        Example:
            >>> upstream = tracker.get_upstream_lineage("fv_customer_features")
            >>> # [
            >>> #     {'node_id': 'TRANSACTIONS', 'type': 'source_table', 'depth': 1},
            >>> #     {'node_id': 'CUSTOMERS', 'type': 'source_table', 'depth': 1}
            >>> # ]
        """
        # Use recursive CTE to traverse graph
        sql = f"""
        WITH RECURSIVE upstream AS (
            -- Base case: start node
            SELECT
                node_id,
                node_type,
                node_name,
                0 AS depth
            FROM {self.database}.{self.schema}.LINEAGE_NODES
            WHERE node_id = ?

            UNION ALL

            -- Recursive case: traverse edges backward
            SELECT
                n.node_id,
                n.node_type,
                n.node_name,
                u.depth + 1 AS depth
            FROM upstream u
            JOIN {self.database}.{self.schema}.LINEAGE_EDGES e
                ON u.node_id = e.target_node_id
            JOIN {self.database}.{self.schema}.LINEAGE_NODES n
                ON e.source_node_id = n.node_id
            WHERE u.depth < ?
        )
        SELECT DISTINCT node_id, node_type, node_name, depth
        FROM upstream
        WHERE depth > 0
        ORDER BY depth, node_name
        """

        result = self.session.sql(sql).bind(node_id, max_depth).collect()

        return [
            {
                "node_id": row["NODE_ID"],
                "type": row["NODE_TYPE"],
                "name": row["NODE_NAME"],
                "depth": row["DEPTH"],
            }
            for row in result
        ]

    def get_downstream_lineage(
        self, node_id: str, max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Get all downstream dependencies (forward lineage).

        This traverses the lineage graph forward to find all nodes
        that depend on the given node.

        Args:
            node_id: ID of the node to trace
            max_depth: Maximum depth to traverse

        Returns:
            List of downstream nodes with their relationships
        """
        sql = f"""
        WITH RECURSIVE downstream AS (
            -- Base case: start node
            SELECT
                node_id,
                node_type,
                node_name,
                0 AS depth
            FROM {self.database}.{self.schema}.LINEAGE_NODES
            WHERE node_id = ?

            UNION ALL

            -- Recursive case: traverse edges forward
            SELECT
                n.node_id,
                n.node_type,
                n.node_name,
                d.depth + 1 AS depth
            FROM downstream d
            JOIN {self.database}.{self.schema}.LINEAGE_EDGES e
                ON d.node_id = e.source_node_id
            JOIN {self.database}.{self.schema}.LINEAGE_NODES n
                ON e.target_node_id = n.node_id
            WHERE d.depth < ?
        )
        SELECT DISTINCT node_id, node_type, node_name, depth
        FROM downstream
        WHERE depth > 0
        ORDER BY depth, node_name
        """

        result = self.session.sql(sql).bind(node_id, max_depth).collect()

        return [
            {
                "node_id": row["NODE_ID"],
                "type": row["NODE_TYPE"],
                "name": row["NODE_NAME"],
                "depth": row["DEPTH"],
            }
            for row in result
        ]

    def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """Analyze the impact of changes to a node.

        This provides a comprehensive view of what would be affected
        if the given node changes.

        Args:
            node_id: ID of the node to analyze

        Returns:
            Dictionary with impact analysis results
        """
        downstream = self.get_downstream_lineage(node_id)

        # Group by type
        impact_by_type: Dict[str, List[str]] = {}
        for node in downstream:
            node_type = node["type"]
            if node_type not in impact_by_type:
                impact_by_type[node_type] = []
            impact_by_type[node_type].append(node["name"])

        return {
            "node_id": node_id,
            "total_impacted": len(downstream),
            "impact_by_type": impact_by_type,
            "impacted_nodes": downstream,
        }
