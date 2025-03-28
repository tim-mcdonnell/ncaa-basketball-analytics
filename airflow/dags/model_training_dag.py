"""
DAG for training NCAA basketball prediction models.
This DAG prepares training data, trains models, evaluates performance, and registers models.
"""

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.data_sensors import DuckDBTableSensor
from airflow.utils.dates import days_ago

# Import model training functions from project
# These would be implemented in the src/models directory
from src.models.data_preparation import prepare_training_data
from src.models.training import train_model
from src.models.evaluation import evaluate_model
from src.models.registry import register_model


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "start_date": days_ago(1),
}

# Database connection settings
conn_id = "duckdb_default"
database = "/path/to/ncaa_basketball.duckdb"  # Update path as needed

# MLflow tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Define the DAG - run weekly since model training is computationally expensive
dag = DAG(
    dag_id="model_training_dag",
    default_args=default_args,
    description="Train NCAA basketball prediction models",
    schedule_interval="0 2 * * 0",  # Run weekly on Sunday at 2 AM
    catchup=False,
    tags=["ncaa", "basketball", "model_training"],
)

with dag:
    # Sensor to check if team features are available
    team_features_available = DuckDBTableSensor(
        task_id="team_features_available",
        conn_id=conn_id,
        database=database,
        table="team_features",
        min_rows=300,  # Ensure we have features for a reasonable number of teams
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60 * 2,  # Timeout after 2 hours
        mode="reschedule",
    )

    # Sensor to check if game features are available
    game_features_available = DuckDBTableSensor(
        task_id="game_features_available",
        conn_id=conn_id,
        database=database,
        table="game_features",
        min_rows=1000,  # Ensure we have features for a reasonable number of games
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60 * 2,  # Timeout after 2 hours
        mode="reschedule",
    )

    # Task to prepare training data
    prepare_training_data_task = PythonOperator(
        task_id="prepare_training_data",
        python_callable=prepare_training_data,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "lookback_days": 365,  # Use data from the last year for training
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
            "output_path": "/path/to/training_data/",  # Update path as needed
        },
        retries=2,
    )

    # Task to train the model
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={
            "input_path": "/path/to/training_data/",  # Update path as needed
            "output_path": "/path/to/trained_models/",  # Update path as needed
            "model_type": "gradient_boosting",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=2,
    )

    # Task to evaluate the model
    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        op_kwargs={
            "model_path": "/path/to/trained_models/",  # Update path as needed
            "test_data_path": "/path/to/training_data/test_data.csv",  # Update path as needed
            "tracking_uri": MLFLOW_TRACKING_URI,
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=1,
    )

    # Task to register the model in the registry
    register_model_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        op_kwargs={
            "model_path": "/path/to/trained_models/",  # Update path as needed
            "model_name": "ncaa_basketball_prediction",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "min_accuracy": 0.70,  # Only register models with at least 70% accuracy
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=1,
    )

    # Define task dependencies - linear flow
    [team_features_available, game_features_available] >> prepare_training_data_task
    prepare_training_data_task >> train_model_task >> evaluate_model_task >> register_model_task
