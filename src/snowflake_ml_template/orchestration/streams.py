"""Stream processing for CDC and event-driven pipelines."""

from snowflake.snowpark import Session

from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


class StreamProcessor:
    """Process change data capture using Snowflake Streams.

    Example:
        >>> processor = StreamProcessor(session, "ML_PROD_DB", "RAW_DATA")
        >>> processor.create_stream("transactions_stream", "TRANSACTIONS")
        >>> processor.process_stream("transactions_stream", process_fn)
    """

    def __init__(self, session: Session, database: str, schema: str):
        """Initialize the stream processor.

        Args:
            session: Active Snowflake session to use for operations
            database: Target database for monitoring
            schema: Target schema for monitoring
        """
        self.session = session
        self.database = database
        self.schema = schema
        self.logger = get_logger(__name__)

    def create_stream(self, stream_name: str, table_name: str) -> bool:
        """Create a stream on a table.

        Args:
            stream_name: Name of the stream to create
            table_name: Name of the table to create the stream on

        Returns:
            True if stream was created successfully, False otherwise
        """
        try:
            sql = f"""
            CREATE OR REPLACE STREAM {self.database}.{self.schema}.{stream_name}
            ON TABLE {self.database}.{self.schema}.{table_name}
            """
            self.session.sql(sql).collect()
            self.logger.info(f"Created stream: {stream_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create stream {stream_name}: {e}")
            return False

    def has_data(self, stream_name: str) -> bool:
        """Check if stream has data.

        Args:
            stream_name: Name of the stream to check

        Returns:
            bool: True if the stream has data, False otherwise or on error
        """
        try:
            result = self.session.sql(
                f"SELECT SYSTEM$STREAM_HAS_DATA('{self.database}.{self.schema}.{stream_name}')"
            ).collect()

            if not result or len(result) == 0:
                return False

            # SYSTEM$STREAM_HAS_DATA() returns 'TRUE' or 'FALSE' as strings
            has_data_str = str(result[0][0]).upper()
            return has_data_str == "TRUE"

        except Exception as e:
            self.logger.error(
                "Failed to check stream",
                extra={"stream_name": stream_name, "error": str(e)},
            )
            return False
