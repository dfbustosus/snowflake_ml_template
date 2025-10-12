"""Helpers to render training stored procedure templates and artifact staging paths."""


def render_training_sproc(
    proc_name: str = "TRAIN_MODEL_PROC", model_stage: str = "@ML_MODELS_STAGE"
) -> str:
    """Return a SQL string defining a stored procedure that trains a model.

    The body is a template and must be customized to the use case. It is
    intentionally generic: it illustrates the pattern (use Snowpark Session,
    write artifact to stage, then log to registry).
    """
    sql = f"""
CREATE OR REPLACE PROCEDURE {proc_name}()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python','snowflake-ml-python')
HANDLER = 'run'
AS
$$
def run(session):
    # Placeholder training flow
    # 1) load data via session.table
    # 2) train model with scikit-learn / ML framework
    # 3) serialize to a file and put to internal stage
    import os
    import joblib

    # TODO: replace with actual training logic
    model = {{'note': 'replace with real model object'}}
    out_file = '/tmp/model.joblib'
    joblib.dump(model, out_file)
    # upload file to internal stage
    stage_path = '{model_stage}/' + os.path.basename(out_file)
    session.file.put(out_file, '{model_stage}', auto_compress=False)
    return stage_path
$$;
"""
    return sql.strip()


def stage_artifact_path(model_name: str, version: str) -> str:
    """Return a canonical path in the internal stage for a model artifact."""
    return f"@ML_MODELS_STAGE/{model_name}/{version}/model.joblib"
