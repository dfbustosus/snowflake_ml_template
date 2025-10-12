-- 05_create_dynamic_table.sql
-- Materialize parsed fields from VARIANT into a structured table
-- NOTE: Dynamic Tables cannot use streams due to change tracking limits.
-- Note: Dynamic tables use Snowflake change-tracking and some SQL linters
-- may not support the DYNAMIC TABLE syntax. We create a regular table
-- here with the same content. To create a dynamic table, run the
-- DYNAMIC TABLE statement manually in Snowflake, e.g.:
--   CREATE OR REPLACE DYNAMIC TABLE creditcard_dynamic
--     TARGET_LAG = '1 HOUR' AS <SELECT ...>;

-- Create the table structure first, then populate it
CREATE OR REPLACE TABLE creditcard_dynamic (
    load_timestamp TIMESTAMP_LTZ,
    event_time FLOAT,
    v1 FLOAT,
    v2 FLOAT,
    v3 FLOAT,
    v4 FLOAT,
    v5 FLOAT,
    v6 FLOAT,
    v7 FLOAT,
    v8 FLOAT,
    v9 FLOAT,
    v10 FLOAT,
    v11 FLOAT,
    v12 FLOAT,
    v13 FLOAT,
    v14 FLOAT,
    v15 FLOAT,
    v16 FLOAT,
    v17 FLOAT,
    v18 FLOAT,
    v19 FLOAT,
    v20 FLOAT,
    v21 FLOAT,
    v22 FLOAT,
    v23 FLOAT,
    v24 FLOAT,
    v25 FLOAT,
    v26 FLOAT,
    v27 FLOAT,
    v28 FLOAT,
    amount FLOAT,
    class INTEGER
);
