# NCAA Basketball Analytics Airflow DAGs

This directory contains Airflow DAGs (Directed Acyclic Graphs) that orchestrate the NCAA basketball prediction system workflows. The DAGs are designed to automate the end-to-end process of data collection, model training, prediction generation, and evaluation.

## DAG Overview

### 1. Data Collection DAG (`data_collection_dag.py`)

This DAG handles the daily collection and ingestion of new NCAA basketball game data:
- Fetches completed game results from external APIs
- Collects team statistics
- Processes and cleans the raw data
- Stores the processed data in the database for use in training and prediction

**Schedule**: Daily at 4:00 AM

### 2. Model Training DAG (`model_training_dag.py`)

This DAG manages the periodic training and registration of prediction models:
- Prepares training data from historical games
- Trains a new prediction model
- Evaluates the model's performance
- Registers the model in MLflow registry if it meets quality criteria

**Schedule**: Weekly on Sundays at midnight

### 3. Prediction DAG (`prediction_dag.py`)

This DAG handles the daily generation of predictions for upcoming games:
- Fetches upcoming NCAA basketball games
- Prepares feature data for prediction
- Generates predictions using the registered model
- Formats and stores prediction results

**Schedule**: Daily at 6:00 AM

### 4. Evaluation DAG (`evaluation_dag.py`)

This DAG evaluates the accuracy of previous predictions:
- Compares predictions to actual game results
- Calculates accuracy and other performance metrics
- Stores evaluation results for monitoring

**Schedule**: Daily at 10:00 AM

## Parameters

Each DAG accepts parameters that can be modified at runtime:

- `duckdb_conn_id`: The Airflow connection ID for DuckDB
- `database_path`: Path to the basketball database
- `output_dir`: Directory for storing output files
- `model_name`: Name of the model in the registry
- Various other parameters specific to each DAG's function

## Dependencies

These DAGs assume the following dependencies:
- The NCAA basketball prediction system codebase
- Airflow with appropriate connections configured
- DuckDB for data storage
- MLflow for model tracking and registry

## Execution Sequence

For a complete prediction workflow:
1. The data collection DAG runs early morning to gather the latest data
2. The prediction DAG runs to generate predictions for upcoming games
3. The evaluation DAG runs later to assess the accuracy of previous predictions
4. On a weekly basis, the model training DAG retrains the model with the latest data

## Error Handling

All DAGs include error handling to:
- Retry failed tasks a configurable number of times
- Send email notifications on failure
- Skip downstream tasks if upstream tasks fail
- Log detailed error information for debugging
