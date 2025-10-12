-- 04_create_stream.sql
-- Create a stream to capture new rows in the raw_creditcard table
CREATE OR REPLACE STREAM raw_creditcard_stream
ON TABLE raw_creditcard
SHOW_INITIAL_ROWS = FALSE;
