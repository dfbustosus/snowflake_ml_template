"""Sphinx configuration for Snowflake ML Template documentation."""

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

# -- Project information -----------------------------------------------------
project = "Snowflake ML Template"
author = "David BU"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Exclude non-existent modules from documentation
autodoc_mock_imports = [
    "pandas",
    "google",
    "google.cloud",
    "google.cloud.bigquery",
    "flytekit",
    "flytekitplugins",
    "pyspark",
    "snowflake",
    "snowflake.snowpark",
    "snowflake.connector",
    "snowflake.ml",
    "snowflake.snowpark.functions",
    "snowflake_ml_python",
    "joblib",
    "lightgbm",
    "xgboost",
    "sklearn",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "pydantic",
    "yaml",
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Intersphinx mapping (optional)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}
