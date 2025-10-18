# ML Pipelines

This directory contains production-ready ML pipelines built on the Golden Snowflake MLOps Framework.

## Architecture

All pipelines follow the Template Method pattern, inheriting from `PipelineTemplate`:

```
BasePipeline (Abstract)
    ↓
PipelineTemplate (Concrete base with framework integration)
    ↓
FraudDetectionPipeline (Model-specific implementation)
```

## Pipeline Structure

Each pipeline implements only two methods:

1. **`engineer_features()`** - Define feature engineering logic
2. **`train_model()`** - Define model training logic

All other steps (infrastructure, validation, deployment, monitoring) are handled by the template.

## Available Pipelines

### 1. Fraud Detection (`fraud_detection/`)

**Purpose**: Detect fraudulent transactions using customer behavior patterns

**Features**:
- Transaction aggregates (count, sum, avg, std)
- Temporal features (7-day, 30-day windows)
- Velocity features (transactions per period)

**Model**: XGBoost Classifier

**Deployment**: Warehouse UDF for batch scoring

**Usage**:
```python
from snowflake_ml_template.pipelines.fraud_detection import FraudDetectionPipeline
from snowflake_ml_template.core.base.pipeline import PipelineConfig

config = PipelineConfig(
    name="fraud_detection",
    version="1.0.0",
    environment="dev",
    database="ML_DEV_DB",
    warehouse="ML_TRAINING_WH"
)

pipeline = FraudDetectionPipeline(session, config)
result = pipeline.execute()
```

### 2. Vehicle Insurance Fraud (`vehicle_insurance_fraud/`)

**Purpose**: Detect fraudulent insurance claims

**Features**: TBD

**Model**: TBD

**Deployment**: TBD

## Creating a New Pipeline

### Step 1: Create Pipeline Class

```python
from snowflake_ml_template.pipelines._base.template import PipelineTemplate

class MyPipeline(PipelineTemplate):
    def engineer_features(self) -> None:
        # Register entities
        entity = self.register_entity(
            name="MY_ENTITY",
            join_keys=["ENTITY_ID"]
        )

        # Create features
        features_df = self.session.table("RAW_DATA.MY_TABLE").select(...)

        # Register feature view
        self.register_feature_view(
            name="my_features",
            entity_names=["MY_ENTITY"],
            feature_df=features_df
        )

    def train_model(self) -> None:
        # Get training data
        training_data = self.session.table("FEATURES.TRAINING_DATA")

        # Configure and train
        trainer = XGBoostTrainer(config)
        self.training_orch.register_trainer("xgboost", trainer)
        result = self.training_orch.execute("xgboost", training_data)

        # Register model
        self.model_registry.register_model(...)
```

### Step 2: Execute Pipeline

```python
config = PipelineConfig(
    name="my_pipeline",
    version="1.0.0",
    environment="dev",
    database="ML_DEV_DB",
    warehouse="ML_TRAINING_WH"
)

pipeline = MyPipeline(session, config)
result = pipeline.execute()
```

## Pipeline Execution Flow

1. **Validation** - Validate configuration
2. **Infrastructure Setup** - Ensure databases/schemas exist
3. **Ingestion** - Load data from sources
4. **Transformation** - Transform raw data
5. **Feature Engineering** - Create features (model-specific)
6. **Training** - Train model (model-specific)
7. **Model Validation** - Validate model performance
8. **Deployment** - Deploy model
9. **Monitoring** - Set up monitoring

## Best Practices

### Feature Engineering
- Use Feature Store for all features
- Register entities before feature views
- Use semantic feature names
- Document feature business logic

### Model Training
- Use appropriate warehouse size
- Log all hyperparameters
- Track metrics in Model Registry
- Use semantic versioning

### Deployment
- Test in DEV before promoting
- Use Warehouse UDF for batch (<1GB)
- Use SPCS for real-time or GPU
- Implement health checks

### Monitoring
- Set up drift detection
- Monitor model performance
- Track data quality
- Configure alerts

## Configuration

Pipeline configuration is managed via `PipelineConfig`:

```python
from snowflake_ml_template.core.base.pipeline import PipelineConfig

config = PipelineConfig(
    name="pipeline_name",           # Pipeline identifier
    version="1.0.0",                # Semantic version
    environment="dev",              # dev, test, or prod
    database="ML_DEV_DB",           # Target database
    warehouse="ML_TRAINING_WH",     # Compute warehouse
    ingestion={...},                # Ingestion config (optional)
    transformation={...},           # Transformation config (optional)
    features={...},                 # Feature config (optional)
    training={...},                 # Training config (optional)
    deployment={...},               # Deployment config (optional)
    monitoring={...}                # Monitoring config (optional)
)
```

## Testing

Test pipelines locally before deployment:

```python
# Unit test individual methods
pipeline = MyPipeline(session, config)
pipeline.engineer_features()
assert "my_features" in pipeline.feature_views

# Integration test full pipeline
result = pipeline.execute()
assert result.status == "success"
```

## Promotion Workflow

1. **DEV**: Develop and test pipeline
2. **TEST**: Validate with production-like data
3. **PROD**: Deploy to production

```python
# Promote model through stages
registry.promote_model("my_model", "1.0.0", ModelStage.TEST)
registry.promote_model("my_model", "1.0.0", ModelStage.PROD)
registry.set_default_version("my_model", "1.0.0")
```

## Troubleshooting

### Pipeline Fails at Feature Engineering
- Check entity registration
- Verify source tables exist
- Validate join keys in data

### Training Fails
- Check warehouse size
- Verify training data quality
- Review hyperparameters

### Deployment Fails
- Verify model artifact exists
- Check deployment permissions
- Validate UDF/service configuration

## Support

For issues or questions:
1. Check pipeline logs
2. Review error messages
3. Consult framework documentation
4. Contact ML platform team
