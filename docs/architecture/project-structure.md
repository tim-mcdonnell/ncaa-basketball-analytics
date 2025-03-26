# Project Structure

## Overview

This document outlines the folder structure and organization for the NCAA Basketball Analytics project. The structure follows modern data engineering and ML practices with clear separation between data pipeline components.

## Directory Structure

The project uses a modular structure that separates code, data, configuration, and documentation:

```
ncaa-basketball-analytics/
├── .github/                      # GitHub workflows and CI/CD configurations
├── airflow/                      # Airflow DAGs and related configurations
├── config/                       # Configuration files
├── data/                         # Data storage directory
├── docs/                         # Project documentation
├── logs/                         # Application logs
├── notebooks/                    # Jupyter notebooks for exploration
├── src/                          # Source code
├── tests/                        # Unit and integration tests
├── app.py                        # Main application entry point
└── pyproject.toml                # Project dependencies and metadata
```

!!! tip "Repository Navigation"
    Understanding the project structure is essential for navigating the codebase. New developers should start by exploring the `src` directory to understand the core functionality.

## Key Directories Explained

### `airflow/`

Contains all Airflow DAGs and related configurations:

```
airflow/
├── dags/                     # DAG definitions
├── plugins/                  # Custom Airflow plugins
└── requirements.txt          # Airflow-specific dependencies
```

DAGs are organized by their function (data ingestion, feature generation, model training) and follow a consistent naming convention.

### `config/`

Stores all configuration files in YAML format:

```
config/
├── api_config.yaml           # ESPN API configuration
├── db_config.yaml            # Database configurations
├── feature_config.yaml       # Feature engineering settings
└── model_config.yaml         # Model training configurations
```

These configurations are validated using Pydantic models defined in `src/config/settings.py`.

### `data/`

Follows a medallion architecture for data organization:

```
data/
├── raw/                      # Raw data from ESPN API (bronze layer)
├── processed/                # Cleansed, transformed data (silver layer)
├── features/                 # Computed features ready for modeling (gold layer)
└── models/                   # Serialized trained models and parameters
```

### `docs/`

Contains comprehensive documentation for the project:

```
docs/
├── architecture/             # System architecture documents
├── guides/                   # How-to guides and tutorials
└── api/                      # API documentation
```

Documentation is built using MkDocs Material, providing searchable and navigable documentation.

### `src/`

Contains all source code for the project, organized by functionality:

```
src/
├── api/                      # ESPN API client code
├── config/                   # Configuration management
├── data/                     # Data processing modules
├── features/                 # Feature engineering code
├── models/                   # Machine learning models
└── visualization/            # Visualization code
```

#### `src/api/`

API client implementation for fetching data from ESPN:

```
src/api/
├── __init__.py
├── client.py                 # API client
└── schemas.py                # API response schemas
```

#### `src/config/`

Configuration management using Pydantic for validation:

```
src/config/
├── __init__.py
└── settings.py               # Pydantic models for config validation
```

#### `src/data/`

Data processing pipelines:

```
src/data/
├── __init__.py
├── ingestion.py              # Data ingestion from API
├── transformation.py         # Data transformation
└── storage.py                # Database interactions
```

#### `src/features/`

Feature engineering framework:

```
src/features/
├── __init__.py
├── base.py                   # Base feature class
├── registry.py               # Feature registry
├── team_features.py          # Team-level features
├── player_features.py        # Player-level features
├── game_features.py          # Game-level features
└── temporal_features.py      # Time-based features
```

#### `src/models/`

Machine learning model training and inference:

```
src/models/
├── __init__.py
├── preprocessing.py          # Data preprocessing for models
├── training.py               # Model training functions
├── evaluation.py             # Model evaluation metrics
└── prediction.py             # Prediction generation
```

#### `src/visualization/`

Plotly Dash components and layouts:

```
src/visualization/
├── __init__.py
├── dashboard.py              # Dash app initialization
└── components/               # Dashboard components
```

### `tests/`

Contains a comprehensive test suite mirroring the structure of the `src` directory:

```
tests/
├── api/                      # API client tests
├── data/                     # Data processing tests
├── features/                 # Feature engineering tests
└── models/                   # Model tests
```

## Development Workflow

The project structure supports the following development workflow:

1. **Configuration**: Update or add configuration in the `config/` directory
2. **Data Collection**: Implement data collection in `src/api/` and `src/data/ingestion.py`
3. **Transformation**: Process raw data in `src/data/transformation.py`
4. **Feature Engineering**: Develop features in `src/features/`
5. **Model Development**: Create and tune models in `src/models/`
6. **Visualization**: Build dashboard components in `src/visualization/`
7. **Orchestration**: Schedule workflows in `airflow/dags/`

## Extension Guidelines

When extending the project:

- **New Features**: Add to appropriate module in `src/features/`
- **New Data Sources**: Create new client in `src/api/` and update ingestion
- **New Models**: Add to `src/models/` with appropriate tests
- **New Visualizations**: Create components in `src/visualization/components/`

!!! note "Implementation Freedom"
    This structure provides a framework for organization, but developers have flexibility in implementation details as long as they maintain the overall architecture.
