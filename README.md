# Golden Snowflake MLOps Framework

<p align="left">
  <a href="https://github.com/dfbustosus/snowflake_ml_template/actions/workflows/pre-commit.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dfbustosus/snowflake_ml_template/pre-commit.yml?label=CI" alt="CI Status" />
  </a>
  <a href="https://github.com/dfbustosus/snowflake_ml_template/actions/workflows/python-lint.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dfbustosus/snowflake_ml_template/python-lint.yml?label=Lint" alt="Lint Status" />
  </a>
  <a href="https://github.com/dfbustosus/snowflake_ml_template/actions/workflows/docs.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dfbustosus/snowflake_ml_template/docs.yml?label=Docs" alt="Docs Status" />
  </a>
  <a href="https://img.shields.io/badge/mypy-checked-2A5DFF">
    <img src="https://img.shields.io/badge/mypy-checked-2A5DFF" alt="Static Typing" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB" alt="Python Versions" />
  <img src="https://img.shields.io/badge/Snowflake-Snowpark-29B5E8" alt="Snowflake Snowpark" />
  <img src="https://img.shields.io/badge/ML_Frameworks-LightGBM%20%7C%20XGBoost%20%7C%20PyTorch%20%7C%20TensorFlow-EF6C00" alt="ML Frameworks" />
  <img src="https://img.shields.io/badge/Orchestration-Snowflake%20Tasks%20%26%20Streams-3C6E71" alt="Orchestration" />
  <img src="https://img.shields.io/badge/Feature%20Store-Point--in--Time%20%7C%20Lineage-6C5CE7" alt="Feature Store" />
  <img src="https://img.shields.io/badge/Monitoring-Data%20%7C%20Model%20%7C%20Infra-A3A1FB" alt="Monitoring" />
  <img src="https://img.shields.io/badge/Deployment-UDF%20%7C%20SPCS-F39C12" alt="Deployment Targets" />
  <img src="https://img.shields.io/badge/Security-Bandit%20%7C%20Safety-DD2C00" alt="Security Tooling" />
  <img src="https://img.shields.io/github/license/dfbustosus/snowflake_ml_template" alt="License" />
</p>

Production-ready MLOps framework for Snowflake with complete automation from data ingestion to model deployment.

## Quick Start

```python
from snowflake_ml_template.pipelines._base.template import PipelineTemplate
from snowflake_ml_template.core.base.pipeline import PipelineConfig

# Define your pipeline
class MyPipeline(PipelineTemplate):
    def engineer_features(self) -> None:
        # Register entities and features
        entity = self.register_entity("CUSTOMER", ["CUSTOMER_ID"])
        features_df = self.session.table("TRANSACTIONS").group_by("CUSTOMER_ID").agg(...)
        self.register_feature_view("customer_features", ["CUSTOMER"], features_df)

    def train_model(self) -> None:
        # Train using any framework
        from snowflake_ml_template.training.frameworks.sklearn_trainer import SklearnTrainer
        trainer = SklearnTrainer(config)
        self.training_orch.register_trainer("sklearn", trainer)
        result = self.training_orch.execute("sklearn", training_data)

# Execute
config = PipelineConfig(name="my_pipeline", version="1.0.0", environment="dev",
                        database="ML_DEV_DB", warehouse="ML_TRAINING_WH")
pipeline = MyPipeline(session, config)
result = pipeline.execute()
```

## Architecture

```
Pipelines → Orchestration → Monitoring → Core Framework → Infrastructure → Feature Store → Snowflake
```

## Features

- **Infrastructure**: Automated provisioning (databases, schemas, roles, warehouses)
- **Feature Store**: Versioning, lineage, point-in-time serving, drift detection
- **Training**: Multi-framework support (XGBoost, sklearn, LightGBM, PyTorch, TensorFlow)
- **Registry**: Semantic versioning, DEV→TEST→PROD promotion
- **Deployment**: Batch (UDF) and real-time (SPCS)
- **Monitoring**: Model performance, data quality, infrastructure health
- **Orchestration**: Automated pipelines with Tasks and Streams

## Installation

```bash
pip install -e .
```

## Configuration

Create `config/prod.yaml`:

```yaml
snowflake:
  account: your_account
  user: your_user
  warehouse: ML_TRAINING_WH
  database: ML_PROD_DB
```

## Documentation

See `/src/snowflake_ml_template/pipelines/README.md` for detailed pipeline documentation.

## Project Structure

```
src/snowflake_ml_template/
├── core/              # Base abstractions, session, config, exceptions
├── infrastructure/    # Database, schema, role, warehouse provisioners
├── feature_store/     # Entity, FeatureView, versioning, serving, monitoring
├── ingestion/         # Data ingestion strategies
├── training/          # Multi-framework training engine
├── registry/          # Model versioning and promotion
├── deployment/        # UDF and SPCS deployment
├── pipelines/         # Pipeline templates and examples
├── orchestration/     # Task and stream orchestration
├── monitoring/        # Global monitoring
└── utils/             # Logging and utilities
```

## Examples

See `src/snowflake_ml_template/pipelines/fraud_detection/` for complete reference implementation.
