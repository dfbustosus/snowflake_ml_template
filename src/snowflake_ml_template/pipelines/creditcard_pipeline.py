"""Pipeline for ingesting and processing the creditcard dataset into Snowflake.

This module provides CreditCardPipeline which provisions DDL, ingests a CSV
into a raw VARIANT table, materializes a dynamic table, and merges data into
the final structured table.
"""

import os
from pathlib import Path

from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkClientException

from snowflake_ml_template.infra.credentials import get_snowflake_session
from snowflake_ml_template.utils.logger import logger


class CreditCardPipeline:
    """Orchestrates ingestion and processing of the creditcard dataset."""

    def __init__(self):
        """Initialize the pipeline and create a Snowpark session."""
        # Initialize Snowpark session
        self.session: Session = get_snowflake_session()
        # Base directory for SQL scripts
        self.root_dir = Path(__file__).parents[3]
        self.ddl_dir = self.root_dir / "scripts" / "snowflake" / "ddl"
        self.pipeline_dir = self.root_dir / "scripts" / "snowflake" / "pipeline"

    def setup_schema(self):
        """Run DDL scripts to create file format, stage, tables, stream, and dynamic table."""
        for ddl_file in sorted(self.ddl_dir.glob("*.sql")):
            sql = ddl_file.read_text()
            # Render simple environment placeholders like ${SNOWFLAKE_WAREHOUSE}
            warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
            if warehouse:
                sql = sql.replace("${SNOWFLAKE_WAREHOUSE}", warehouse)
            logger.info(f"Applying DDL: {ddl_file.name}")
            # Some SQL files can contain multiple statements; execute each separately
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                try:
                    logger.debug(
                        f"Executing statement: {stmt[:120]}{'...' if len(stmt)>120 else ''}"
                    )
                    self.session.sql(stmt).collect()
                except Exception as e:
                    logger.error(f"Failed executing statement in {ddl_file.name}: {e}")
                    raise

    def ingest_raw(self, local_file_path: str):
        """Upload a local CSV file into Snowflake stage and copy into the raw table."""
        stage_path = "@ml_raw_stage"
        logger.info(f"Uploading {local_file_path} to stage {stage_path}")
        try:
            # PUT local file to stage (with overwrite for repeated runs)
            self.session.file.put(local_file_path, stage_path, overwrite=True)
        except SnowparkClientException as e:
            logger.warning(f"PUT failed or already exists: {e}")

        # Run COPY INTO raw table (this will skip duplicates due to FORCE=FALSE)
        copy_sql = (self.pipeline_dir / "01_copy_into_raw.sql").read_text()
        logger.info("Copying data into raw_creditcard")
        self.session.sql(copy_sql).collect()

    def process_stream(self):
        """Process new rows from the dynamic table and merge into the final table."""
        merge_sql = (self.pipeline_dir / "02_process_stream.sql").read_text()
        logger.info("Merging from stream into creditcard table")
        self.session.sql(merge_sql).collect()

    def run_full_pipeline(self, local_file_path: str):
        """Execute full pipeline: setup schema, ingest data, and process stream."""
        self.setup_schema()
        self.ingest_raw(local_file_path)
        self.process_stream()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run CreditCard data pipeline on Snowflake"
    )
    parser.add_argument("csv", help="Path to local creditcard CSV file")
    args = parser.parse_args()

    pipeline = CreditCardPipeline()
    pipeline.run_full_pipeline(args.csv)
