# Development Setup

This document describes how to set up a local development environment for the Customer Journey UA project.

## Snowflake Sessions

Be sure to define `SNOWFLAKE_REGION`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, and `SNOWFLAKE_DATABASE` in your environment variables before running the pipeline.

### Feature Store client attachment

The Feature Store refactor expects the active Snowpark session to expose a `feature_store` attribute that implements the Snowflake Feature Store client interface (for example, `snowflake.ml.feature_store.FeatureStore`). When creating sessions outside managed notebooks, attach the client explicitly:

```python
from snowflake.ml.feature_store import FeatureStore as SfFeatureStore
from snowflake.snowpark import Session

session = Session.builder.configs(connection_parameters).create()
FeatureStore.attach_feature_store_client(session, SfFeatureStore(session))

fs = FeatureStore(session=session, database="ML_DEV_DB", schema="FEATURES")
```

### Governance configuration

`FeatureStore` accepts an optional `governance` dictionary to automatically apply Snowflake tags and masking policies whenever feature views are registered. Tags must be fully qualified, and masking policies are referenced by name:

```python
governance = {
    "feature_views": {
        "CUSTOMER_FEATURES": {
            "tags": {
                "governance.data_classification": "CONFIDENTIAL",
                "team.owner": "ml-platform"
            },
            "masking_policies": {
                "SENSITIVE_EMAIL": "SECURITY_SCHEMA.EMAIL_MASK"
            }
        }
    }
}

fs = FeatureStore(
    session=session,
    database="ML_DEV_DB",
    schema="FEATURES",
    governance=governance,
)
```

When `CUSTOMER_FEATURES` is registered, the store emits the corresponding `ALTER ... SET TAG` and `ALTER ... MODIFY COLUMN ... SET MASKING POLICY` statements.

### Monitoring persistence and scheduling

Quality and drift monitors can persist metrics/results into Snowflake tables and schedule recurring checks using Snowflake tasks:

```python
quality_monitor = FeatureQualityMonitor(
    session,
    database="ML_DEV_DB",
    schema="FEATURE_MONITORING",
    table="FEATURE_QUALITY_EVENTS",
)

drift_detector = FeatureDriftDetector(
    session,
    database="ML_DEV_DB",
    schema="FEATURE_MONITORING",
    table="FEATURE_DRIFT_EVENTS",
)

# After computing metrics/results
quality_monitor.record_metrics(metrics, feature_view="CUSTOMER_FEATURES", run_id="2025-10-18")
drift_detector.record_result(result, feature_view="CUSTOMER_FEATURES", run_id="2025-10-18")

# Optional: create Snowflake tasks backing stored procedures
quality_monitor.create_quality_task(
    task_name="ML_DEV_DB.FEATURE_MONITORING.QUALITY_TASK",
    warehouse="MONITORING_WH",
    schedule="USING CRON */30 * * * * UTC",
    procedure_call="CALL FEATURE_MONITORING.RUN_QUALITY_CHECK()",
)

drift_detector.create_drift_task(
    task_name="ML_DEV_DB.FEATURE_MONITORING.DRIFT_TASK",
    warehouse="MONITORING_WH",
    schedule="USING CRON */30 * * * * UTC",
    procedure_call="CALL FEATURE_MONITORING.RUN_DRIFT_CHECK()",
)
```

Both monitors automatically create the backing tables if they do not exist. Implement stored procedures to orchestrate the data collection before scheduling tasks.

### Feature view version metadata

`FeatureVersionManager` now reads Snowflake's metadata view (`SNOWFLAKE.ML.FEATURE_STORE.FEATURE_VIEW_VERSIONS`). Version creation and deprecation must be performed via the Snowflake Feature Store API or Snowsight, but listing versions and computing diffs remain available:

```python
version_manager = FeatureVersionManager(session, "ML_DEV_DB", "FEATURES")

available_versions = version_manager.list_versions("CUSTOMER_FEATURES")
diff = version_manager.compare_versions("CUSTOMER_FEATURES", "1.0.0", "1.1.0")
```

Prerequisites
- Python 3.10+ (3.10 recommended)
- Git
- Adequate memory for data processing (8GB+ recommended)

Create a virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Run pre-commit hooks locally

```bash
pre-commit install
pre-commit run --all-files
```

Running tests

```bash
# Run all unit tests
pytest -v
```

<!-- Running the Flyte workflow locally (dev)

```bash
pyflyte run src/snowflake_ml_template/workflows/customer_journey_workflow.py customer_journey_workflow
``` -->

Current coverage areas:
- URL utilities (normalization, validation)
- Trie data structure (insertion, search, matching)
- MySQL database manager (connection, queries, error handling)
- BigQuery manager (queries, table operations)
- Date utilities (month calculations, lookback windows)
- Data pipeline integration (URL matching, location application)
- Task validation and error propagation
- Workflow data flow and configuration

Documentation

```bash
# Documentation commands
python -m pip install --upgrade pip
python -m pip install -r docs/requirements.txt

# Run pydocstyle checks
pydocstyle src/snowflake_ml_template
# Build Sphinx docs (normal) - run inside the docs directory
cd docs
sphinx-build -b html . _build/html

# Build Sphinx docs and treat warnings as errors - run inside the docs directory
cd docs
sphinx-build -b html -W . _build/html

# Clean build directory
rm -rf docs/_build/html
mkdir -p docs/_build/html

# Open generated docs in default browser (macOS)
cd docs/_build/html
open index.html

# Serve generated docs locally
cd docs/_build/html
python -m http.server 8000
```
