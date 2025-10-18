"""Infrastructure health monitoring."""

from typing import Any, Dict

from snowflake.snowpark import Session

from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class InfrastructureMonitor:
    """Monitor infrastructure health and costs."""

    def __init__(self, session: Session):
        """Initialize the infrastructure monitor.

        Args:
            session: Active Snowflake session to use for operations
        """
        self.session = session
        self.logger = get_logger(__name__)

    def get_warehouse_usage(self, warehouse: str) -> Dict[str, Any]:
        """Get warehouse usage statistics.

        Args:
            warehouse: Name of the warehouse to check

        Returns:
            Dictionary containing warehouse usage metrics
        """
        sql = f"""
        SELECT SUM(credits_used) as total_credits
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
        WHERE warehouse_name = '{warehouse}'
        AND start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
        """
        result = self.session.sql(sql).collect()
        return {
            "warehouse": warehouse,
            "credits_7d": float(result[0][0]) if result else 0.0,
        }
