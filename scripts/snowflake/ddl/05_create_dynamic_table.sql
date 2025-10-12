-- 05_create_dynamic_table.sql
-- Materialize parsed fields from VARIANT into a structured table
-- NOTE: Dynamic Tables cannot use streams due to change tracking limits.
-- Note: Dynamic tables use Snowflake change-tracking and some SQL linters
-- may not support the DYNAMIC TABLE syntax. We create a regular table
-- here with the same content. To create a dynamic table, run the
-- DYNAMIC TABLE statement manually in Snowflake, e.g.:
--   CREATE OR REPLACE DYNAMIC TABLE creditcard_dynamic
--     TARGET_LAG = '1 HOUR' AS <SELECT ...>;
CREATE OR REPLACE TABLE creditcard_dynamic AS
SELECT
    load_timestamp,
    TRY_CAST(
        parsed:Time::STRING AS
        FLOAT
    ) AS event_time,
    TRY_CAST(
        parsed:V1::STRING AS
        FLOAT
    ) AS v1,
    TRY_CAST(
        parsed:V2::STRING AS
        FLOAT
    ) AS v2,
    TRY_CAST(
        parsed:V3::STRING AS
        FLOAT
    ) AS v3,
    TRY_CAST(
        parsed:V4::STRING AS
        FLOAT
    ) AS v4,
    TRY_CAST(
        parsed:V5::STRING AS
        FLOAT
    ) AS v5,
    TRY_CAST(
        parsed:V6::STRING AS
        FLOAT
    ) AS v6,
    TRY_CAST(
        parsed:V7::STRING AS
        FLOAT
    ) AS v7,
    TRY_CAST(
        parsed:V8::STRING AS
        FLOAT
    ) AS v8,
    TRY_CAST(
        parsed:V9::STRING AS
        FLOAT
    ) AS v9,
    TRY_CAST(
        parsed:V10::STRING AS
        FLOAT
    ) AS v10,
    TRY_CAST(
        parsed:V11::STRING AS
        FLOAT
    ) AS v11,
    TRY_CAST(
        parsed:V12::STRING AS
        FLOAT
    ) AS v12,
    TRY_CAST(
        parsed:V13::STRING AS
        FLOAT
    ) AS v13,
    TRY_CAST(
        parsed:V14::STRING AS
        FLOAT
    ) AS v14,
    TRY_CAST(
        parsed:V15::STRING AS
        FLOAT
    ) AS v15,
    TRY_CAST(
        parsed:V16::STRING AS
        FLOAT
    ) AS v16,
    TRY_CAST(
        parsed:V17::STRING AS
        FLOAT
    ) AS v17,
    TRY_CAST(
        parsed:V18::STRING AS
        FLOAT
    ) AS v18,
    TRY_CAST(
        parsed:V19::STRING AS
        FLOAT
    ) AS v19,
    TRY_CAST(
        parsed:V20::STRING AS
        FLOAT
    ) AS v20,
    TRY_CAST(
        parsed:V21::STRING AS
        FLOAT
    ) AS v21,
    TRY_CAST(
        parsed:V22::STRING AS
        FLOAT
    ) AS v22,
    TRY_CAST(
        parsed:V23::STRING AS
        FLOAT
    ) AS v23,
    TRY_CAST(
        parsed:V24::STRING AS
        FLOAT
    ) AS v24,
    TRY_CAST(
        parsed:V25::STRING AS
        FLOAT
    ) AS v25,
    TRY_CAST(
        parsed:V26::STRING AS
        FLOAT
    ) AS v26,
    TRY_CAST(
        parsed:V27::STRING AS
        FLOAT
    ) AS v27,
    TRY_CAST(
        parsed:V28::STRING AS
        FLOAT
    ) AS v28,
    TRY_CAST(
        parsed:Amount::STRING AS
        FLOAT
    ) AS amount,
    TRY_CAST(
        parsed:Class::STRING AS
        INTEGER
    ) AS class
FROM (
    SELECT
        load_timestamp,
        PARSE_JSON(content) AS parsed
    FROM raw_creditcard
);
