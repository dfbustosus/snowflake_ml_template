# Development Setup

This document describes how to set up a local development environment for the Customer Journey UA project.

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
