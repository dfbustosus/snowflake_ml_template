-- 02_create_stage.sql
-- Create an internal stage for local PUT operations (used by session.file.put).
-- If you want to use external cloud storage + Snowpipe, replace this with
-- an external stage with URL and configure AUTO_REFRESH / notifications
-- in your cloud provider (AWS S3, Azure Blob, GCS).
CREATE OR REPLACE STAGE ml_raw_stage
    FILE_FORMAT = csv_format;
