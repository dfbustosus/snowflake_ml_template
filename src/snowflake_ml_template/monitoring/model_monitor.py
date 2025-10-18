"""Model performance monitoring."""

from dataclasses import dataclass

from snowflake.snowpark import Session

from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float


class ModelMonitor:
    """Monitor model performance in production.

    Example:
        >>> monitor = ModelMonitor(session, "ML_PROD_DB", "MONITORING")
        >>> metrics = monitor.get_model_metrics("fraud_detector", "1.0.0")
        >>> if metrics.accuracy < 0.9:
        ...     monitor.trigger_alert("Model performance degraded")
    """

    def __init__(self, session: Session, database: str, schema: str):
        """Initialize the model monitor.

        Args:
            session: Active Snowflake session to use for operations
            database: Target database for monitoring
            schema: Target schema for monitoring
        """
        self.session = session
        self.database = database
        self.schema = schema
        self.logger = get_logger(__name__)
        self._ensure_monitoring_tables()

    def _ensure_monitoring_tables(self) -> None:
        """Create monitoring tables."""
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.MODEL_METRICS (
            model_name VARCHAR,
            version VARCHAR,
            timestamp TIMESTAMP_NTZ,
            accuracy FLOAT,
            precision FLOAT,
            recall FLOAT,
            f1_score FLOAT,
            auc FLOAT
        )
        """
        self.session.sql(sql).collect()

    def log_metrics(self, metrics: ModelMetrics) -> None:
        """Log model metrics."""
        sql = f"""
        INSERT INTO {self.database}.{self.schema}.MODEL_METRICS
        VALUES (?, ?, CURRENT_TIMESTAMP(), ?, ?, ?, ?, ?)
        """
        self.session.sql(sql).bind(
            metrics.model_name,
            metrics.version,
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1_score,
            metrics.auc,
        ).collect()
        self.logger.info(f"Logged metrics for {metrics.model_name} v{metrics.version}")
