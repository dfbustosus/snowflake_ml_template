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

    def ingest_raw(self, local_file_path: str):
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
                    'Time', SPLIT_PART($1, ',', 1),
                    'V1', SPLIT_PART($1, ',', 2),
                    'V2', SPLIT_PART($1, ',', 3),
                    'V3', SPLIT_PART($1, ',', 4),
                    'V4', SPLIT_PART($1, ',', 5),
                    'V5', SPLIT_PART($1, ',', 6),
                    'V6', SPLIT_PART($1, ',', 7),
                    'V7', SPLIT_PART($1, ',', 8),
                    'V8', SPLIT_PART($1, ',', 9),
                    'V9', SPLIT_PART($1, ',', 10),
                    'V10', SPLIT_PART($1, ',', 11),
                    'V11', SPLIT_PART($1, ',', 12),
                    'V12', SPLIT_PART($1, ',', 13),
                    'V13', SPLIT_PART($1, ',', 14),
                    'V14', SPLIT_PART($1, ',', 15),
                    'V15', SPLIT_PART($1, ',', 16),
                    'V16', SPLIT_PART($1, ',', 17),
                    'V17', SPLIT_PART($1, ',', 18),
                    'V18', SPLIT_PART($1, ',', 19),
                    'V19', SPLIT_PART($1, ',', 20),
                    'V20', SPLIT_PART($1, ',', 21),
                    'V21', SPLIT_PART($1, ',', 22),
                    'V22', SPLIT_PART($1, ',', 23),
                    'V23', SPLIT_PART($1, ',', 24),
                    'V24', SPLIT_PART($1, ',', 25),
                    'V25', SPLIT_PART($1, ',', 26),
                    'V26', SPLIT_PART($1, ',', 27),
                    'V27', SPLIT_PART($1, ',', 28),
                    'V28', SPLIT_PART($1, ',', 29),
                    'Amount', SPLIT_PART($1, ',', 30),
                    'Class', SPLIT_PART($1, ',', 31)
                )::VARIANT AS content
            FROM @ml_raw_stage
        )
        FILE_FORMAT = csv_format
        """
        logger.info("Copying data into raw_creditcard")
        self.session.sql(copy_sql).collect()

    def process_stream(self):
        """Process new rows from the dynamic table and merge into the final table."""
        # First populate the dynamic table from raw data
        logger.info("Populating dynamic table from raw data")
        populate_sql = """
        INSERT INTO creditcard_dynamic
        SELECT
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
        )
        WHERE load_timestamp NOT IN (SELECT load_timestamp FROM creditcard_dynamic)
        """
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
