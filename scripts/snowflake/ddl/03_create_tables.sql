-- 03_create_raw_table.sql
-- Create raw landing table for credit card dataset
CREATE OR REPLACE TABLE raw_creditcard (
    content VARIANT,
    load_timestamp TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP ()
);
