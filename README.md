# Golden Snowflake MLOps Framework

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
