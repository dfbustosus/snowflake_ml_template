# Snowflake ML Template

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/snowflake-ml-template.svg)](https://pypi.org/project/snowflake-ml-template/)
[![Coverage](https://codecov.io/gh/dfbustosus/snowflake_ml_template/branch/main/graph/badge.svg)](https://codecov.io/gh/dfbustosus/snowflake_ml_template)
[![Documentation](https://readthedocs.org/projects/snowflake-ml-template/badge/?version=latest)](https://snowflake-ml-template.readthedocs.io/en/latest/)

A comprehensive template for building scalable ML pipelines on Snowflake. This project provides a production-ready framework for machine learning workflows, including data ingestion, model training, deployment, and monitoring.

## 🚀 Features

- **Snowflake Integration**: Native support for Snowflake's Snowpark and ML capabilities
- **Scalable ML Pipelines**: Built-in support for distributed training and inference
- **Data Quality**: Integrated Great Expectations for data validation
- **Experiment Tracking**: MLflow integration for model versioning and monitoring
- **MLOps**: Complete CI/CD pipelines with automated testing and deployment
- **Interactive Dashboards**: Streamlit apps for model monitoring and analysis
- **Type Safety**: Full type hints with mypy support
- **Security**: Automated security scanning and dependency checks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.9 or higher
- Snowflake account with appropriate permissions
- Git

### Install from PyPI

```bash
pip install snowflake-ml-template
```

### Install from Source

```bash
git clone https://github.com/dfbustosus/snowflake_ml_template.git
cd snowflake_ml_template
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/dfbustosus/snowflake_ml_template.git
cd snowflake_ml_template
pip install -e ".[dev]"
pre-commit install
```

## 🚀 Quick Start

1. **Set up your environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your Snowflake credentials
   ```

2. **Run the CLI**:
   ```bash
   snowflake-ml --help
   ```

3. **Train a sample model**:
   ```bash
   python -m snowflake_ml_template.cli train --config config/sample_config.yaml
   ```

## 📁 Project Structure

```
snowflake_ml_template/
├── cli/                    # Command-line interface
├── config/                 # Configuration files
│   └── environments/       # Environment-specific configs
├── data/                   # Data processing modules
├── infra/                  # Infrastructure as code
├── ingest/                 # Data ingestion pipelines
├── models/                 # ML model definitions
│   └── registry/           # Model registry
├── monitoring/             # Monitoring and alerting
├── utils/                  # Utility functions
│   └── session/            # Snowflake session management
└── tests/                  # Unit and integration tests
```

## ⚙️ Configuration

The project uses YAML configuration files for different environments. Copy and modify the templates:

```yaml
# config/environments/development.yaml
snowflake:
  account: your-account
  user: your-user
  warehouse: your-warehouse
  database: your-database
  schema: your-schema

mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: snowflake_ml_experiment

data_quality:
  expectations_path: expectations/
  validation_results_path: validation_results/
```

## 📖 Usage

### Data Ingestion

```python
from snowflake_ml_template.ingest import DataIngestor

ingestor = DataIngestor(config_path="config/environments/production.yaml")
ingestor.ingest_data(source_table="RAW_DATA", target_table="PROCESSED_DATA")
```

### Model Training

```python
from snowflake_ml_template.models import ModelTrainer

trainer = ModelTrainer(config_path="config/environments/production.yaml")
model = trainer.train_model(
    features=["feature1", "feature2", "feature3"],
    target="target_column",
    model_type="random_forest"
)
```

### Model Deployment

```python
from snowflake_ml_template.models.registry import ModelRegistry

registry = ModelRegistry(config_path="config/environments/production.yaml")
registry.deploy_model(model, model_name="my_model", version="1.0.0")
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=snowflake_ml_template

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
pytest -m "snowflake"  # Requires Snowflake connection
```

## 🚀 Deployment

### Local Development

```bash
# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

# Start Streamlit dashboard
streamlit run src/snowflake_ml_template/monitoring/dashboard.py
```

### Production Deployment

The project includes GitHub Actions workflows for:

- **CI/CD**: Automated testing and deployment
- **Infrastructure**: Terraform configurations for Snowflake resources
- **Documentation**: Sphinx documentation builds
- **Security**: Automated security scanning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for API changes
- Ensure all CI checks pass

## 📊 Monitoring & Observability

- **MLflow**: Experiment tracking and model registry
- **Great Expectations**: Data quality monitoring
- **Custom Dashboards**: Streamlit applications for model performance
- **Logging**: Structured logging with structlog

## 🔒 Security

This project includes automated security scanning:

- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability scanning
- **Pre-commit hooks**: Code quality checks

## 📚 Documentation

Full documentation is available at [https://snowflake-ml-template.readthedocs.io/](https://snowflake-ml-template.readthedocs.io/)

Build documentation locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Snowflake for providing the ML platform
- The open-source community for the amazing tools and libraries
- Contributors and maintainers of this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/dfbustosus/snowflake_ml_template/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dfbustosus/snowflake_ml_template/discussions)
- **Documentation**: [Read the Docs](https://snowflake-ml-template.readthedocs.io/)

---

Made with ❤️ for the Snowflake ML community
