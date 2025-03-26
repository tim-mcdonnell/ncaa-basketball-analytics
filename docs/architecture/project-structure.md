# NCAA Basketball Analytics Project Structure

## Overview

This document outlines the folder structure and organization for the NCAA Basketball Analytics project. The structure follows modern data engineering and ML practices with clear separation between data pipeline components.

## Root Directory Structure

```
ncaa-basketball-analytics/
├── .github/                      # GitHub workflows and CI/CD configurations
├── airflow/                      # Airflow DAGs and related configurations
│   ├── dags/                     # DAG definitions
│   ├── plugins/                  # Custom Airflow plugins
│   └── requirements.txt          # Airflow-specific dependencies
├── config/                       # Configuration files
│   ├── api_config.yaml           # ESPN API configuration
│   ├── db_config.yaml            # Database configurations
│   ├── feature_config.yaml       # Feature engineering settings
│   └── model_config.yaml         # Model training configurations
├── data/                         # Data storage directory
│   ├── raw/                      # Raw data from ESPN API
│   ├── processed/                # Processed and transformed data
│   ├── features/                 # Feature sets
│   └── models/                   # Trained models
├── docs/                         # Project documentation
│   ├── architecture/             # System architecture documents
│   │   ├── project-structure.md      # Project structure document
│   │   ├── config-management.md      # Configuration management
│   │   ├── tech-stack.md             # Tech stack overview
│   │   ├── airflow-workflows.md      # Airflow workflow definitions
│   │   ├── data-table-structures.md  # Database schema design
│   │   ├── logging-strategy.md       # Logging implementation
│   │   ├── feature-engineering.md    # Feature engineering framework
│   │   └── model-training.md         # Model training pipeline
│   ├── guides/                   # How-to guides and tutorials
│   │   ├── getting-started.md        # Setup and initial usage
│   │   ├── adding-features.md        # How to add new features
│   │   └── training-models.md        # How to train new models
│   └── api/                      # API documentation
│       └── espn-endpoints.md         # ESPN API endpoints reference
├── logs/                         # Application logs
├── notebooks/                    # Jupyter notebooks for exploration and analysis
├── src/                          # Source code
│   ├── api/                      # ESPN API client code
│   │   ├── __init__.py
│   │   ├── client.py             # API client
│   │   └── schemas.py            # API response schemas
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py           # Pydantic models for config validation
│   ├── data/                     # Data processing modules
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Data ingestion from API
│   │   ├── transformation.py     # Data transformation
│   │   └── storage.py            # Database interactions
│   ├── features/                 # Feature engineering code
│   │   ├── __init__.py
│   │   ├── base.py               # Base feature class
│   │   ├── registry.py           # Feature registry
│   │   ├── team_features.py      # Team-level features
│   │   ├── player_features.py    # Player-level features
│   │   ├── game_features.py      # Game-level features
│   │   └── temporal_features.py  # Time-based features
│   ├── models/                   # Machine learning models
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # Data preprocessing for models
│   │   ├── training.py           # Model training functions
│   │   ├── evaluation.py         # Model evaluation metrics
│   │   └── prediction.py         # Prediction generation
│   └── visualization/            # Visualization code
│       ├── __init__.py
│       ├── dashboard.py          # Dash app initialization
│       └── components/           # Dashboard components
│           ├── __init__.py
│           ├── team_view.py      # Team-specific visuals
│           ├── game_view.py      # Game-specific visuals
│           └── prediction_view.py # Prediction visualizations
├── tests/                        # Unit and integration tests
│   ├── api/                      # API client tests
│   ├── data/                     # Data processing tests
│   ├── features/                 # Feature engineering tests
│   └── models/                   # Model tests
├── app.py                        # Main Dash application entry point
├── .env.example                  # Example environment variables
├── .gitignore                    # Git ignore file
├── pyproject.toml                # Project dependencies and metadata
├── README.md                     # Project overview documentation
└── setup.py                      # Package installation script
```

## Key Components Explained

### airflow/
Contains all Airflow DAGs and related configurations. DAGs are organized by their function (data ingestion, feature generation, model training).

### config/
Stores all configuration files in YAML format. These are validated using Pydantic models defined in `src/config/settings.py`.

### data/
Follows a medallion architecture:
- `raw/`: Raw JSON data from ESPN API (bronze layer)
- `processed/`: Cleansed, transformed data (silver layer)
- `features/`: Computed features ready for modeling (gold layer)
- `models/`: Serialized trained models and parameters

### src/
Contains all source code for the project:

- `api/`: ESPN API client implementation
- `config/`: Configuration management using Pydantic
- `data/`: Data processing pipelines
- `features/`: Feature engineering framework
- `models/`: ML model training and inference
- `visualization/`: Plotly Dash components and layouts

### tests/
Contains comprehensive test suite mirroring the structure of the `src` directory.
