"""Pipeline for ingesting and processing the creditcard dataset into Snowflake.

This module provides CreditCardPipeline which provisions DDL, ingests a CSV
into a raw VARIANT table, materializes a dynamic table, and merges data into
the final structured table.
"""

import hashlib
import os
from pathlib import Path

from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkClientException

from snowflake_ml_template.infra.credentials import get_snowflake_session
from snowflake_ml_template.utils.logger import logger


class CreditCardPipeline:
    """Orchestrates ingestion and processing of the creditcard dataset."""

    def __init__(self) -> None:
        """Initialize the pipeline and create a Snowpark session."""
        # Initialize Snowpark session
        self.session: Session = get_snowflake_session()
        # Base directory for SQL scripts
        self.root_dir = Path(__file__).parents[3]
        self.ddl_dir = self.root_dir / "scripts" / "snowflake" / "ddl"
        self.pipeline_dir = self.root_dir / "scripts" / "snowflake" / "pipeline"

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file to detect changes."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _is_file_already_processed(self, file_path: str) -> bool:
        """Check if this exact file has already been processed by comparing hashes."""
        file_hash = self._get_file_hash(file_path)

        # Check if we have a processed_files table, create if not exists
        self.session.sql(
            """
            CREATE TABLE IF NOT EXISTS processed_files (
                file_path STRING,
                file_hash STRING,
                processed_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
                PRIMARY KEY (file_path)
            )
        """
        ).collect()

        # Check if this file hash already exists
        result = self.session.sql(  # nosec
            f"""
            SELECT COUNT(*) as count
            FROM processed_files
            WHERE file_hash = '{file_hash}'
            """
        ).collect()
        return bool(result[0]["COUNT"]) if result else False

    def _mark_file_as_processed(self, file_path: str) -> None:
        """Mark a file as processed by storing its hash."""
        file_hash = self._get_file_hash(file_path)

        # Insert or update the processed file record
        self.session.sql(  # nosec
            f"""
            MERGE INTO processed_files target
            USING (SELECT '{file_path}' as file_path, '{file_hash}' as file_hash) source
            ON target.file_path = source.file_path
            WHEN MATCHED THEN
                UPDATE SET file_hash = source.file_hash, processed_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (file_path, file_hash) VALUES (source.file_path, source.file_hash)
            """
        ).collect()

    def setup_schema(self) -> None:
        """Run DDL scripts to create file format, stage, tables, stream, and dynamic table."""
        # Ensure database and schema exist
        database = os.getenv("SNOWFLAKE_DATABASE")
        schema = os.getenv("SNOWFLAKE_SCHEMA")
        if database and schema:
            logger.info(f"Ensuring database {database} and schema {schema} exist")
            self.session.sql(f"CREATE DATABASE IF NOT EXISTS {database}").collect()
            self.session.sql(
                f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}"
            ).collect()
            self.session.sql(f"USE DATABASE {database}").collect()
            self.session.sql(f"USE SCHEMA {schema}").collect()

        # Skip the test DDL file that might cause issues
        ddl_files = [
            f
            for f in sorted(self.ddl_dir.glob("*.sql"))
            if not f.name.startswith("00_test")
        ]

        for ddl_file in ddl_files:
            sql = ddl_file.read_text()
            # Render simple environment placeholders like ${SNOWFLAKE_WAREHOUSE}
            warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
            if warehouse:
                sql = sql.replace("${SNOWFLAKE_WAREHOUSE}", warehouse)
            logger.info(f"Applying DDL: {ddl_file.name}")

            # Execute the entire SQL file as one statement first
            try:
                logger.debug(f"Executing DDL file: {ddl_file.name}")
                self.session.sql(sql).collect()
                logger.info(f"Successfully applied DDL: {ddl_file.name}")
            except Exception as e:
                logger.warning(f"Failed to execute DDL file as single statement: {e}")
                # Fall back to statement-by-statement execution
                # Split by semicolon and clean up statements
                raw_statements = sql.split(";")
                statements = []
                for stmt in raw_statements:
                    # Remove comments and empty lines
                    lines = [
                        line.strip()
                        for line in stmt.split("\n")
                        if line.strip() and not line.strip().startswith("--")
                    ]
                    cleaned_stmt = "\n".join(lines).strip()
                    if cleaned_stmt:
                        statements.append(cleaned_stmt)

                logger.info(
                    f"Falling back to executing {len(statements)} statements individually"
                )

                for i, stmt in enumerate(statements, 1):
                    if not stmt:
                        continue
                    try:
                        logger.debug(
                            f"Executing statement {i}: {stmt[:100]}{'...' if len(stmt)>100 else ''}"
                        )
                        self.session.sql(stmt).collect()
                    except Exception as stmt_e:
                        logger.error(
                            f"Failed executing statement {i} in {ddl_file.name}: {stmt_e}"
                        )
                        logger.error(f"Statement was: {stmt}")
                        raise

    def ingest_raw(self, local_file_path: str) -> None:
        """Upload a local CSV file into Snowflake stage and copy into the raw table."""
        stage_path = "@ml_raw_stage"
        logger.info(f"Uploading {local_file_path} to stage {stage_path}")
        try:
            # PUT local file to stage (with overwrite for repeated runs)
            self.session.file.put(local_file_path, stage_path, overwrite=True)
        except SnowparkClientException as e:
            logger.warning(f"PUT failed: {e}")
            raise

        # Run COPY INTO raw table - embedded SQL to avoid file issues
        copy_sql = """
        COPY INTO raw_creditcard (content)
        FROM (
            SELECT
                OBJECT_CONSTRUCT(
                    'Time', REPLACE(REPLACE(SPLIT_PART($1, ',', 1), '"', ''), '"', ''),
                    'V1', REPLACE(REPLACE(SPLIT_PART($1, ',', 2), '"', ''), '"', ''),
                    'V2', REPLACE(REPLACE(SPLIT_PART($1, ',', 3), '"', ''), '"', ''),
                    'V3', REPLACE(REPLACE(SPLIT_PART($1, ',', 4), '"', ''), '"', ''),
                    'V4', REPLACE(REPLACE(SPLIT_PART($1, ',', 5), '"', ''), '"', ''),
                    'V5', REPLACE(REPLACE(SPLIT_PART($1, ',', 6), '"', ''), '"', ''),
                    'V6', REPLACE(REPLACE(SPLIT_PART($1, ',', 7), '"', ''), '"', ''),
                    'V7', REPLACE(REPLACE(SPLIT_PART($1, ',', 8), '"', ''), '"', ''),
                    'V8', REPLACE(REPLACE(SPLIT_PART($1, ',', 9), '"', ''), '"', ''),
                    'V9', REPLACE(REPLACE(SPLIT_PART($1, ',', 10), '"', ''), '"', ''),
                    'V10', REPLACE(REPLACE(SPLIT_PART($1, ',', 11), '"', ''), '"', ''),
                    'V11', REPLACE(REPLACE(SPLIT_PART($1, ',', 12), '"', ''), '"', ''),
                    'V12', REPLACE(REPLACE(SPLIT_PART($1, ',', 13), '"', ''), '"', ''),
                    'V13', REPLACE(REPLACE(SPLIT_PART($1, ',', 14), '"', ''), '"', ''),
                    'V14', REPLACE(REPLACE(SPLIT_PART($1, ',', 15), '"', ''), '"', ''),
                    'V15', REPLACE(REPLACE(SPLIT_PART($1, ',', 16), '"', ''), '"', ''),
                    'V16', REPLACE(REPLACE(SPLIT_PART($1, ',', 17), '"', ''), '"', ''),
                    'V17', REPLACE(REPLACE(SPLIT_PART($1, ',', 18), '"', ''), '"', ''),
                    'V18', REPLACE(REPLACE(SPLIT_PART($1, ',', 19), '"', ''), '"', ''),
                    'V19', REPLACE(REPLACE(SPLIT_PART($1, ',', 20), '"', ''), '"', ''),
                    'V20', REPLACE(REPLACE(SPLIT_PART($1, ',', 21), '"', ''), '"', ''),
                    'V21', REPLACE(REPLACE(SPLIT_PART($1, ',', 22), '"', ''), '"', ''),
                    'V22', REPLACE(REPLACE(SPLIT_PART($1, ',', 23), '"', ''), '"', ''),
                    'V23', REPLACE(REPLACE(SPLIT_PART($1, ',', 24), '"', ''), '"', ''),
                    'V24', REPLACE(REPLACE(SPLIT_PART($1, ',', 25), '"', ''), '"', ''),
                    'V25', REPLACE(REPLACE(SPLIT_PART($1, ',', 26), '"', ''), '"', ''),
                    'V26', REPLACE(REPLACE(SPLIT_PART($1, ',', 27), '"', ''), '"', ''),
                    'V27', REPLACE(REPLACE(SPLIT_PART($1, ',', 28), '"', ''), '"', ''),
                    'V28', REPLACE(REPLACE(SPLIT_PART($1, ',', 29), '"', ''), '"', ''),
                    'Amount', REPLACE(REPLACE(SPLIT_PART($1, ',', 30), '"', ''), '"', ''),
                    'Class', REPLACE(REPLACE(SPLIT_PART($1, ',', 31), '"', ''), '"', '')
                )::VARIANT AS content
            FROM @ml_raw_stage
        )
        FILE_FORMAT = csv_format
        """
        logger.info("Copying data into raw_creditcard")
        self.session.sql(copy_sql).collect()

    def process_stream(self) -> None:
        """Process new rows from the dynamic table and merge into the final table."""
        # Get the latest load_timestamp from raw table
        latest_ts_result = self.session.sql(
            "SELECT MAX(load_timestamp) as latest FROM raw_creditcard"
        ).collect()
        if not latest_ts_result or not latest_ts_result[0]["LATEST"]:
            logger.warning("No data found in raw_creditcard table")
            return

        latest_load_timestamp = latest_ts_result[0]["LATEST"]
        logger.info(f"Processing data from load timestamp: {latest_load_timestamp}")

        # Clear dynamic table first to ensure clean state
        self.session.sql("DELETE FROM creditcard_dynamic").collect()

        # First populate the dynamic table from raw data - only from the latest load
        populate_sql = f"""
        INSERT INTO creditcard_dynamic
        SELECT DISTINCT
            load_timestamp,
            TRY_CAST(parsed:Time::STRING AS FLOAT) AS event_time,
            TRY_CAST(parsed:V1::STRING AS FLOAT) AS v1,
            TRY_CAST(parsed:V2::STRING AS FLOAT) AS v2,
            TRY_CAST(parsed:V3::STRING AS FLOAT) AS v3,
            TRY_CAST(parsed:V4::STRING AS FLOAT) AS v4,
            TRY_CAST(parsed:V5::STRING AS FLOAT) AS v5,
            TRY_CAST(parsed:V6::STRING AS FLOAT) AS v6,
            TRY_CAST(parsed:V7::STRING AS FLOAT) AS v7,
            TRY_CAST(parsed:V8::STRING AS FLOAT) AS v8,
            TRY_CAST(parsed:V9::STRING AS FLOAT) AS v9,
            TRY_CAST(parsed:V10::STRING AS FLOAT) AS v10,
            TRY_CAST(parsed:V11::STRING AS FLOAT) AS v11,
            TRY_CAST(parsed:V12::STRING AS FLOAT) AS v12,
            TRY_CAST(parsed:V13::STRING AS FLOAT) AS v13,
            TRY_CAST(parsed:V14::STRING AS FLOAT) AS v14,
            TRY_CAST(parsed:V15::STRING AS FLOAT) AS v15,
            TRY_CAST(parsed:V16::STRING AS FLOAT) AS v16,
            TRY_CAST(parsed:V17::STRING AS FLOAT) AS v17,
            TRY_CAST(parsed:V18::STRING AS FLOAT) AS v18,
            TRY_CAST(parsed:V19::STRING AS FLOAT) AS v19,
            TRY_CAST(parsed:V20::STRING AS FLOAT) AS v20,
            TRY_CAST(parsed:V21::STRING AS FLOAT) AS v21,
            TRY_CAST(parsed:V22::STRING AS FLOAT) AS v22,
            TRY_CAST(parsed:V23::STRING AS FLOAT) AS v23,
            TRY_CAST(parsed:V24::STRING AS FLOAT) AS v24,
            TRY_CAST(parsed:V25::STRING AS FLOAT) AS v25,
            TRY_CAST(parsed:V26::STRING AS FLOAT) AS v26,
            TRY_CAST(parsed:V27::STRING AS FLOAT) AS v27,
            TRY_CAST(parsed:V28::STRING AS FLOAT) AS v28,
            TRY_CAST(parsed:Amount::STRING AS FLOAT) AS amount,
            TRY_CAST(parsed:Class::STRING AS INTEGER) AS class
        FROM (
            SELECT
                load_timestamp,
                PARSE_JSON(content) AS parsed
            FROM raw_creditcard
            WHERE load_timestamp = '{latest_load_timestamp}'
        )
        """  # nosec
        self.session.sql(populate_sql).collect()

        # Then run the merge - embedded SQL to avoid file issues
        merge_sql = """
        MERGE INTO creditcard target
        USING creditcard_dynamic source
        ON target.event_time = source.event_time AND target.load_timestamp = source.load_timestamp
        WHEN MATCHED THEN
            UPDATE SET
                target.v1 = source.v1,
                target.v2 = source.v2,
                target.v3 = source.v3,
                target.v4 = source.v4,
                target.v5 = source.v5,
                target.v6 = source.v6,
                target.v7 = source.v7,
                target.v8 = source.v8,
                target.v9 = source.v9,
                target.v10 = source.v10,
                target.v11 = source.v11,
                target.v12 = source.v12,
                target.v13 = source.v13,
                target.v14 = source.v14,
                target.v15 = source.v15,
                target.v16 = source.v16,
                target.v17 = source.v17,
                target.v18 = source.v18,
                target.v19 = source.v19,
                target.v20 = source.v20,
                target.v21 = source.v21,
                target.v22 = source.v22,
                target.v23 = source.v23,
                target.v24 = source.v24,
                target.v25 = source.v25,
                target.v26 = source.v26,
                target.v27 = source.v27,
                target.v28 = source.v28,
                target.amount = source.amount,
                target.class = source.class
        WHEN NOT MATCHED THEN
            INSERT (
                event_time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                v21, v22, v23, v24, v25, v26, v27, v28, amount, class, load_timestamp
            )
            VALUES (
                source.event_time, source.v1, source.v2, source.v3, source.v4, source.v5,
                source.v6, source.v7, source.v8, source.v9, source.v10, source.v11,
                source.v12, source.v13, source.v14, source.v15, source.v16, source.v17,
                source.v18, source.v19, source.v20, source.v21, source.v22, source.v23,
                source.v24, source.v25, source.v26, source.v27, source.v28, source.amount,
                source.class, source.load_timestamp
            )
        """
        logger.info("Merging from dynamic table into creditcard table")
        self.session.sql(merge_sql).collect()

    def run_full_pipeline(
        self, local_file_path: str, incremental: bool = False
    ) -> None:
        """Execute full pipeline: setup schema, ingest data, and process stream.

        Args:
            local_file_path: Path to the CSV file to process
            incremental: If True, skip schema setup and only process new data
        """
        # Check if file has already been processed
        if self._is_file_already_processed(local_file_path):
            logger.info(
                f"File {local_file_path} has already been processed. Skipping pipeline."
            )
            return

        logger.info(f"Processing new file: {local_file_path}")

        if not incremental:
            self.setup_schema()

        self.ingest_raw(local_file_path)
        self.process_stream()

        # Mark file as processed
        self._mark_file_as_processed(local_file_path)
        logger.info(f"Successfully processed and marked file: {local_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run CreditCard data pipeline on Snowflake"
    )
    parser.add_argument("csv", help="Path to local creditcard CSV file")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip schema setup and only process new data (requires existing tables)",
    )
    args = parser.parse_args()

    pipeline = CreditCardPipeline()
    pipeline.run_full_pipeline(args.csv, incremental=args.incremental)
