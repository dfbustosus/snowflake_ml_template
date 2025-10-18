"""snowflake_ml_template.

Lightweight package providing SQL templates, helpers and scaffolding for
implementing the "Golden Migration Plan" for migrating ML workloads to Snowflake.
This package intentionally avoids importing snowflake modules at import time
so that unit tests and static analysis run without a Snowflake connection.
"""

__all__ = [
    "core",
    "deployment",
    "feature_store",
    "infrastructure",
    "ingestion",
    "monitoring",
    "orchestration",
    "pipelines",
    "registry",
    "training",
    "utils",
]
