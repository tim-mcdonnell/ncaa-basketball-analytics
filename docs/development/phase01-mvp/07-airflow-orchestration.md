---
title: Airflow Orchestration
description: Technical specification for Airflow orchestration in Phase 01 MVP
---

# Airflow Orchestration

This document provides technical details for implementing the Airflow orchestration component of Phase 01 MVP.

## 🎯 Overview

**Background:** Reliable workflow orchestration is critical for consistently running data pipelines, ensuring data freshness, and maintaining model accuracy over time.

**Objective:** Establish a reliable workflow management system for scheduling and monitoring data pipelines.

**Scope:** This component will orchestrate the end-to-end process from data collection to model training and prediction generation using Apache Airflow.

## 📐 Technical Requirements

### Architecture

```
airflow/
├── dags/                     # DAG definitions
│   ├── __init__.py
│   ├── data_collection_dag.py # Data collection pipeline
│   ├── feature_engineering_dag.py # Feature calculation pipeline
│   ├── model_training_dag.py # Model training pipeline
│   └── prediction_dag.py    # Prediction generation pipeline
├── operators/                # Custom operators
│   ├── __init__.py
│   └── espn_operators.py     # ESPN API operators
├── hooks/                    # Custom hooks
│   ├── __init__.py
│   └── duckdb_hook.py        # DuckDB connection hook
├── sensors/                  # Custom sensors
│   ├── __init__.py
│   └── data_sensors.py       # Data availability sensors
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── logging_utils.py      # Logging utilities
│   └── alerting_utils.py     # Alerting utilities
└── config/                   # Airflow configuration
    ├── airflow.cfg           # Airflow configuration file
    └── variables.json        # Airflow variables
```

### DAG Design

1. Data Collection DAG must:
   - Efficiently collect data from ESPN API endpoints
   - Support incremental data collection
   - Implement appropriate scheduling (daily/weekly)
   - Handle errors and retries
   - Log collection results
   - Support async API requests
   - Implement adaptive rate limiting

2. Feature Engineering DAG must:
   - Calculate features based on collected data
   - Handle feature dependencies
   - Support incremental feature updates
   - Track feature computation metrics
   - Validate feature quality

3. Model Training DAG must:
   - Prepare training datasets
   - Train models with appropriate hyperparameters
   - Track model performance metrics
   - Register trained models
   - Handle model evaluation
   - Support model versioning

4. Prediction DAG must:
   - Generate predictions for upcoming games
   - Store predictions in appropriate tables
   - Track prediction accuracy over time
   - Support multiple prediction models

### Custom Components

1. Custom operators must:
   - Implement ESPN API data collection
   - Support DuckDB integration
   - Enable feature computation
   - Facilitate model training

2. Custom hooks must:
   - Provide consistent database access
   - Manage API authentication
   - Support efficient data retrieval

3. Custom sensors must:
   - Check for data availability
   - Monitor system health
   - Detect data quality issues

### Workflow Management

1. Task dependencies must:
   - Reflect logical data flow
   - Support parallel execution where possible
   - Implement appropriate wait conditions
   - Handle cross-DAG dependencies

2. Error handling must:
   - Implement appropriate retry policies
   - Log detailed error information
   - Provide alerting for critical failures
   - Support graceful degradation

3. Monitoring must:
   - Track task execution times
   - Monitor resource usage
   - Provide execution history
   - Support debugging of failed tasks

## 🧪 Testing Requirements

### Test-Driven Development Process

1. **RED Phase**:
   - Write failing tests for DAG structure and dependencies
   - Create tests for custom operators and hooks
   - Develop tests for task execution and error handling
   - Write tests for end-to-end workflow execution
   - Create tests for scheduling and triggers

2. **GREEN Phase**:
   - Implement DAG structures that pass dependency tests
   - Build custom operators and hooks that satisfy tests
   - Create error handling that correctly manages failures
   - Develop end-to-end workflows that pass execution tests
   - Implement scheduling that meets trigger requirements

3. **REFACTOR Phase**:
   - Optimize task execution efficiency
   - Enhance error handling robustness
   - Improve monitoring and observability
   - Refine DAG structure for clarity
   - Optimize resource usage

### Test Cases

- [ ] Test `test_dag_structure`: Verify DAG structure and dependencies are correct
- [ ] Test `test_dag_loading`: Verify all DAGs load without errors
- [ ] Test `test_operator_execution`: Verify custom operators execute correctly
- [ ] Test `test_duckdb_hook`: Verify DuckDB hook connects and queries correctly
- [ ] Test `test_espn_operator`: Verify ESPN operator collects data correctly
- [ ] Test `test_data_sensor`: Verify data availability sensors function properly
- [ ] Test `test_error_handling`: Verify error handling and retry logic
- [ ] Test `test_cross_dag_dependencies`: Verify cross-DAG dependencies resolve correctly
- [ ] Test `test_incremental_data_collection`: Verify incremental collection works
- [ ] Test `test_feature_calculation`: Verify feature calculation tasks work correctly
- [ ] Test `test_model_training`: Verify model training tasks execute properly
- [ ] Test `test_prediction_generation`: Verify prediction tasks generate correct outputs
- [ ] Test `test_end_to_end_workflow`: Verify full workflow executes correctly

### DAG Testing Example

```python
def test_data_collection_dag_structure():
    """
    Test that the data collection DAG has the correct structure.
    """
    # Import the DAG
    from airflow.dags.data_collection_dag import data_collection_dag
    
    # Get the DAG
    dag = data_collection_dag
    
    # Assert DAG properties
    assert dag.dag_id == "data_collection"
    assert dag.schedule_interval == "0 4 * * *"  # Daily at 4:00 AM
    assert dag.catchup is False
    
    # Get tasks
    collect_teams = dag.get_task("collect_teams")
    collect_games = dag.get_task("collect_games")
    collect_players = dag.get_task("collect_players")
    collect_stats = dag.get_task("collect_stats")
    data_quality = dag.get_task("data_quality")
    
    # Assert task dependencies
    assert collect_games.upstream_list == [collect_teams]
    assert collect_players.upstream_list == [collect_games]
    assert collect_stats.upstream_list == [collect_players]
    assert data_quality.upstream_list == [collect_stats]
```

### Operator Testing Example

```python
def test_espn_teams_operator():
    """
    Test that the ESPN teams operator correctly collects team data.
    """
    # Arrange
    from airflow.operators.espn_operators import ESPNTeamsOperator
    import os
    
    # Create a test connection
    conn_id = "espn_api_test"
    
    # Create the operator
    operator = ESPNTeamsOperator(
        task_id="test_collect_teams",
        conn_id=conn_id,
        duckdb_path=":memory:",
        timeout=30,
        retries=3
    )
    
    # Act
    # Execute the operator
    operator.execute(context={})
    
    # Assert
    # Check that teams were collected
    from src.data.storage.db import get_connection
    conn = get_connection(":memory:")
    result = conn.execute("SELECT COUNT(*) FROM raw_teams").fetchone()
    
    # Verify we have teams data
    assert result[0] > 0, "No teams data was collected"
```

### Real-World Testing

- Run: `airflow dags test data_collection_dag 2023-01-01`
- Verify: 
  1. All tasks execute without errors
  2. Data is collected and stored correctly
  3. Appropriate logs are generated

- Run: `airflow tasks test data_collection_dag collect_teams 2023-01-01`
- Verify:
  1. Task executes successfully
  2. Team data is collected and stored
  3. Appropriate logs are generated

## 📄 Documentation Requirements

- [ ] Create DAG documentation in `docs/guides/airflow-dags.md`
- [ ] Document custom operators in `docs/guides/airflow-operators.md`
- [ ] Document error handling strategy in `docs/guides/airflow-error-handling.md`
- [ ] Create workflow diagrams in `docs/architecture/airflow-workflows.md`
- [ ] Add Airflow setup guide in `docs/guides/airflow-setup.md`

### Code Documentation Standards

- All DAGs must have:
  - Module-level docstrings explaining the purpose and schedule
  - Default arguments documentation
  - Task documentation

- All custom operators, hooks, and sensors must have:
  - Class-level docstrings explaining purpose
  - Method documentation with parameters and return values
  - Usage examples

## 🛠️ Implementation Process

1. Set up local Airflow environment
2. Implement DuckDB hook for database access
3. Create ESPN API operators for data collection
4. Develop data availability sensors
5. Implement data collection DAG
6. Create feature engineering DAG
7. Develop model training DAG
8. Implement prediction generation DAG
9. Add monitoring and alerting utilities
10. Configure error handling and retry policies

## ✅ Acceptance Criteria

- [ ] All specified tests pass, including end-to-end workflow tests
- [ ] Data collection DAG successfully retrieves and stores data
- [ ] Feature engineering DAG correctly calculates features
- [ ] Model training DAG successfully trains and evaluates models
- [ ] Prediction DAG generates accurate predictions
- [ ] Error handling correctly manages failures and retries
- [ ] Monitoring provides visibility into workflow execution
- [ ] Cross-DAG dependencies are correctly managed
- [ ] Scheduling executes workflows at appropriate times
- [ ] Documentation completely describes the Airflow implementation
- [ ] Code meets project quality standards (passes linting and typing)

## DAG Specifications

### Data Collection DAG

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  collect_teams  ├────►│  collect_games  ├────►│ collect_players │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ collect_stats   │
                                               │                 │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │  data_quality   │
                                               │                 │
                                               └─────────────────┘
```

- **Schedule**: Daily at 4:00 AM
- **Concurrency**: 4 task instances
- **Timeout**: 2 hours
- **Retries**: 3 with exponential backoff

### Feature Engineering DAG

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ check_new_data  ├────►│ team_features   ├────►│ player_features │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ game_features   │
                                               │                 │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ feature_quality │
                                               │                 │
                                               └─────────────────┘
```

- **Schedule**: Daily at 6:00 AM (after data collection)
- **Concurrency**: 2 task instances
- **Timeout**: 1 hour
- **Retries**: 2 with exponential backoff

### Model Training DAG

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ check_features  ├────►│ prepare_dataset ├────►│  train_model    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ evaluate_model  │
                                               │                 │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ register_model  │
                                               │                 │
                                               └─────────────────┘
```

- **Schedule**: Weekly on Sunday at 2:00 AM
- **Concurrency**: 1 task instance
- **Timeout**: 4 hours
- **Retries**: 1 with exponential backoff

### Prediction DAG

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  check_model    ├────►│ get_upcoming    ├────►│ generate_preds  │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │  store_preds    │
                                               │                 │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ evaluate_preds  │
                                               │                 │
                                               └─────────────────┘
```

- **Schedule**: Daily at 8:00 AM
- **Concurrency**: 1 task instance
- **Timeout**: 30 minutes
- **Retries**: 3 with exponential backoff

## Custom Operator Implementation Example

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from src.data.api.espn_client import AsyncESPNClient
from src.data.storage.db import get_connection
import logging
import asyncio

class ESPNTeamsOperator(BaseOperator):
    """
    Operator that collects team data from the ESPN API.
    
    This operator uses the AsyncESPNClient to efficiently collect
    team data and store it in the DuckDB database.
    """
    
    @apply_defaults
    def __init__(
        self,
        conn_id,
        duckdb_path,
        timeout=60,
        retries=3,
        *args,
        **kwargs
    ):
        """
        Initialize the operator.
        
        Args:
            conn_id: Airflow connection ID for ESPN API
            duckdb_path: Path to DuckDB database
            timeout: Request timeout in seconds
            retries: Number of retries for API requests
        """
        super().__init__(*args, **kwargs)
        self.conn_id = conn_id
        self.duckdb_path = duckdb_path
        self.timeout = timeout
        self.retries = retries
    
    def execute(self, context):
        """
        Execute the operator.
        
        Args:
            context: Airflow task execution context
            
        Returns:
            Number of teams collected
        """
        logging.info("Collecting team data from ESPN API")
        
        # Get API connection details from Airflow connection
        conn = self.get_connection(self.conn_id)
        api_key = conn.password
        
        # Create API client
        async def fetch_teams():
            async with AsyncESPNClient(
                api_key=api_key,
                timeout=self.timeout,
                retries=self.retries
            ) as client:
                return await client.get_teams()
        
        # Execute async client
        teams = asyncio.run(fetch_teams())
        
        # Store in database
        db_conn = get_connection(self.duckdb_path)
        for team in teams:
            db_conn.execute(
                """
                INSERT INTO raw_teams (team_id, raw_data, source_url, processing_version)
                VALUES (?, ?, ?, ?)
                """,
                (
                    team["id"],
                    team.json(),
                    "https://api.espn.com/v1/sports/basketball/mens-college-basketball/teams",
                    "1.0.0"
                )
            )
        
        logging.info(f"Collected {len(teams)} teams")
        return len(teams)
```

## Architecture Alignment

This Airflow orchestration implementation aligns with the specifications in the architecture documentation:

1. Uses Apache Airflow as specified in tech-stack.md
2. Follows the workflow architecture outlined in airflow-workflows.md
3. Implements proper task dependencies and scheduling
4. Integrates with DuckDB and API components
5. Supports the end-to-end data pipeline and workflow
6. Enables monitoring and observability

## Integration Points

- **Data Collection**: Orchestrates API data collection
- **Feature Engineering**: Schedules feature computation
- **Model Training**: Manages model training and evaluation
- **Prediction**: Schedules prediction generation
- **Monitoring**: Integrates with logging and alerting systems

## Technical Challenges

1. **Workflow Dependencies**: Managing complex dependencies between tasks
2. **Error Handling**: Implementing robust error handling and recovery
3. **Data Validation**: Ensuring data quality throughout the pipeline
4. **Scheduling Optimizations**: Balancing frequency with resource usage
5. **Cross-DAG Dependencies**: Managing dependencies between separate DAGs

## Success Metrics

1. **Reliability**: >99% workflow completion rate
2. **Timeliness**: Workflows complete within defined time windows
3. **Resource Efficiency**: Optimal resource utilization during execution
4. **Observability**: Complete visibility into workflow status and history
5. **Maintainability**: Modular design that supports easy updates 