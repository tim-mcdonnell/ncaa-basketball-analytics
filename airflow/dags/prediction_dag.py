"""
DAG for generating NCAA basketball game predictions.
This DAG fetches upcoming games, prepares prediction data, generates predictions, and stores results.
"""

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.data_sensors import DuckDBTableSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable

# Import prediction functions from project
# These would be implemented in the src/predictions directory
from src.predictions.upcoming_games import fetch_upcoming_games
from src.predictions.data_preparation import prepare_prediction_data
from src.predictions.prediction import generate_predictions
from src.predictions.storage import store_predictions


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "start_date": days_ago(1),
}

# Load configuration from Airflow variables
database = Variable.get("ncaa_basketball_db_path")
data_dir = Variable.get("ncaa_basketball_data_dir")
predictions_dir = Variable.get("ncaa_basketball_predictions_dir")
mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
forecast_days = int(Variable.get("prediction_forecast_days"))

# Database connection settings
conn_id = "duckdb_default"

# Define the DAG - run daily for predictions
dag = DAG(
    dag_id="prediction_dag",
    default_args=default_args,
    description="Generate NCAA basketball game predictions",
    schedule_interval="0 8 * * *",  # Run daily at 8 AM
    catchup=False,
    tags=["ncaa", "basketball", "prediction"],
)

with dag:
    # Sensor to check if team features are available
    team_features_available = DuckDBTableSensor(
        task_id="team_features_available",
        conn_id=conn_id,
        database=database,
        table="team_features",
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60,  # Timeout after 1 hour
        mode="reschedule",
    )

    # Sensor to check if game features are available
    game_features_available = DuckDBTableSensor(
        task_id="game_features_available",
        conn_id=conn_id,
        database=database,
        table="game_features",
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60,  # Timeout after 1 hour
        mode="reschedule",
    )

    # Task to fetch upcoming games
    fetch_upcoming_games_task = PythonOperator(
        task_id="fetch_upcoming_games",
        python_callable=fetch_upcoming_games,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "days_ahead": forecast_days,  # Fetch games for the configured forecast days
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=3,
    )

    # Task to prepare prediction data
    prepare_prediction_data_task = PythonOperator(
        task_id="prepare_prediction_data",
        python_callable=prepare_prediction_data,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "output_path": f"{data_dir}/prediction_data/",
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=2,
    )

    # Task to generate predictions
    generate_predictions_task = PythonOperator(
        task_id="generate_predictions",
        python_callable=generate_predictions,
        op_kwargs={
            "input_path": f"{data_dir}/prediction_data/",
            "output_path": f"{predictions_dir}/",
            "model_stage": "production",  # Use production model from registry
            "tracking_uri": mlflow_tracking_uri,
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=2,
    )

    # Task to store predictions in the database
    store_predictions_task = PythonOperator(
        task_id="store_predictions",
        python_callable=store_predictions,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "predictions_path": f"{predictions_dir}/",
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=2,
    )

    # Define task dependencies - linear flow
    [team_features_available, game_features_available] >> fetch_upcoming_games_task
    (
        fetch_upcoming_games_task
        >> prepare_prediction_data_task
        >> generate_predictions_task
        >> store_predictions_task
    )
