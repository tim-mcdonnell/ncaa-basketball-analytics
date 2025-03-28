"""
DAG for training the NCAA basketball prediction model.
This DAG prepares training data, trains the model, evaluates its performance, and registers it.
"""

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.data_sensors import DuckDBTableSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable

# Import model training functions from project
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

# Load configuration from Airflow variables
database = Variable.get("ncaa_basketball_db_path")
data_dir = Variable.get("ncaa_basketball_data_dir")
models_dir = Variable.get("ncaa_basketball_models_dir")
mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
lookback_days = int(Variable.get("model_training_lookback_days"))
min_accuracy = float(Variable.get("min_accuracy_threshold"))

# Database connection settings
conn_id = "duckdb_default"

# Define the DAG - run weekly for model training
dag = DAG(
    dag_id="model_training_dag",
    default_args=default_args,
    description="Train the NCAA basketball prediction model",
    schedule_interval="0 2 * * 0",  # Run weekly at 2 AM on Sunday
    catchup=False,
    tags=["ncaa", "basketball", "model", "training"],
)

with dag:
    # Sensor to check if team features are available
    team_features_available = DuckDBTableSensor(
        task_id="team_features_available",
        conn_id=conn_id,
        database=database,
        table="team_features",
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
            "output_path": f"{data_dir}/training_data/",
            "lookback_days": lookback_days,  # Use data from the past year
            "test_split": 0.2,  # Use 20% of data for validation
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
    )

    # Task to train the model
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={
            "input_path": f"{data_dir}/training_data/",
            "output_path": f"{models_dir}/{{ execution_date.strftime('%Y-%m-%d') }}/",
            "model_type": "gradient_boosting",  # Use gradient boosting model
            "tracking_uri": mlflow_tracking_uri,
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
    )

    # Task to evaluate the model
    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        op_kwargs={
            "model_path": f"{models_dir}/{{ execution_date.strftime('%Y-%m-%d') }}/model.pt",
            "test_data_path": f"{data_dir}/training_data/test_data.parquet",
            "output_path": f"{models_dir}/{{ execution_date.strftime('%Y-%m-%d') }}/evaluation/",
            "tracking_uri": mlflow_tracking_uri,
            "run_id": "{{ task_instance.xcom_pull(task_ids='train_model')['run_id'] }}",
            "min_accuracy_threshold": min_accuracy,  # Minimum required accuracy
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
    )

    # Task to register the model
    register_model_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        op_kwargs={
            "model_path": f"{models_dir}/{{ execution_date.strftime('%Y-%m-%d') }}/model.pt",
            "config_path": f"{models_dir}/{{ execution_date.strftime('%Y-%m-%d') }}/model_config.json",
            "evaluation_path": f"{models_dir}/{{ execution_date.strftime('%Y-%m-%d') }}/evaluation/metrics.json",
            "tracking_uri": mlflow_tracking_uri,
            "model_name": "ncaa_basketball_prediction",
            "stage": "production",  # Register as production if performance is good
            "run_id": "{{ task_instance.xcom_pull(task_ids='train_model')['run_id'] }}",
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
    )

    # Define task dependencies
    [team_features_available, game_features_available] >> prepare_training_data_task
    prepare_training_data_task >> train_model_task >> evaluate_model_task >> register_model_task
