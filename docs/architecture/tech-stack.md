# Tech Stack

## Overview

This document outlines the technology stack for the NCAA Basketball Analytics project, detailing each component's role and integration points. The technologies were selected to create a scalable, maintainable analytics platform optimized for sports data.

```mermaid
graph TD
    A[Python 3.12] --> B[DuckDB]
    A --> C[Polars]
    A --> D[Apache Airflow]
    A --> E[Plotly Dash]
    B <--> C
    C --> E
    D --> B
```

## Core Technologies

### Python 3.12

**Role**: Core programming language for all system components.

**Benefits**:
- Latest stable Python version with performance improvements
- Type annotation support for better code quality
- Pattern matching for cleaner data parsing
- Rich ecosystem of data science and machine learning libraries

### Apache Airflow

**Role**: Workflow orchestration platform for scheduling and monitoring data pipelines.

**Key Features**:
- DAGs (Directed Acyclic Graphs) for workflow definition
- Scheduling with cron-like expressions
- Error handling and automatic retries
- Web UI for monitoring and management

**Integration Points**:
- Orchestrates data collection from ESPN API
- Schedules feature computation and model training
- Manages incremental vs. full recalculations

!!! example "Example DAG Structure"
    ```python
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime, timedelta

    with DAG(
        'espn_data_collection',
        schedule=timedelta(days=1),
        start_date=datetime(2023, 1, 1),
        catchup=False,
    ) as dag:
        
        fetch_teams = PythonOperator(
            task_id='fetch_teams',
            python_callable=fetch_teams_data,
        )
        
        fetch_games = PythonOperator(
            task_id='fetch_games',
            python_callable=fetch_games_data,
        )
        
        fetch_teams >> fetch_games
    ```

### DuckDB

**Role**: Analytical database for storing, querying, and transforming sports data.

**Key Features**:
- Column-oriented storage optimized for analytics
- SQL interface with advanced analytical functions
- Direct integration with Polars and Pandas
- In-process architecture with minimal setup

**Integration Points**:
- Primary storage for processed data
- Computation engine for complex queries
- Integration with feature engineering framework

!!! example "Example Query"
    ```sql
    -- Calculate team performance over a rolling window
    SELECT 
        team_id,
        game_date,
        points,
        AVG(points) OVER (
            PARTITION BY team_id 
            ORDER BY game_date 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) as avg_points_last_5
    FROM team_game_stats
    ```

### Polars

**Role**: Data manipulation library for efficient transformations and feature engineering.

**Key Features**:
- High-performance operations on large datasets
- Memory efficiency with lazy evaluation
- Native support for parallel execution
- Seamless integration with DuckDB

**Integration Points**:
- Primary data transformation library
- Feature engineering computations
- Preprocessing for ML models

!!! example "Example Usage"
    ```python
    import polars as pl
    
    # Feature engineering with window functions
    features_df = (
        df.group_by("team_id")
        .sort("game_date")
        .select([
            pl.col("team_id"),
            pl.col("game_id"),
            pl.col("points"),
            pl.col("points").mean().over(
                "team_id", 
                pl.col("game_date").rolling_window(5)
            ).alias("avg_points_5")
        ])
    )
    ```

### Plotly Dash

**Role**: Interactive web dashboard for data visualization and exploration.

**Key Features**:
- Interactive charts and graphs
- React-based components with Python backend
- Callback system for dynamic updates
- Layout customization with Bootstrap

**Integration Points**:
- Frontend for data exploration
- Visualization of model predictions
- Interactive team and player analysis

## Supporting Libraries

The following libraries provide additional functionality:

| Library | Purpose | Integration |
|---------|---------|-------------|
| **Pydantic** | Data validation and settings management | Configuration validation |
| **LightGBM** | Gradient boosting framework | Primary ML algorithm |
| **PyTorch** | Deep learning framework | Alternative ML models |
| **MLflow** | ML experiment tracking | Model versioning and monitoring |
| **Pandas** | Data manipulation (legacy support) | Data integration and export |
| **pytest** | Testing framework | Unit and integration tests |

## Development Tools

The development environment is supported by:

- **Poetry**: Dependency management
- **pre-commit**: Automated code quality checks
- **MkDocs Material**: Documentation generation
- **GitHub Actions**: CI/CD pipelines

## Technology Selection Criteria

Technologies were selected based on the following criteria:

1. **Performance**: Ability to handle large volumes of sports data efficiently
2. **Maintainability**: Clear APIs and good documentation
3. **Community Support**: Active development and user community
4. **Flexibility**: Adaptable to changing requirements
5. **Simplicity**: Prefer simpler solutions when possible

!!! note "Technology Evolution"
    This tech stack represents the current state of the project. Technologies may be added or replaced as requirements evolve and new tools emerge.
