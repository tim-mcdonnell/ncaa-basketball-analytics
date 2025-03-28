"""
DAG for feature engineering of NCAA basketball data.
This DAG calculates team, player, and game features for prediction models.
"""

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.data_sensors import DuckDBTableSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable

# Import feature calculation functions
from src.features.team_features import calculate_team_features
from src.features.player_features import calculate_player_features
from src.features.game_features import calculate_game_features


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
lookback_days = int(Variable.get("feature_engineering_lookback_days"))

# Database connection settings
conn_id = "duckdb_default"

# Define the DAG - run daily for feature engineering
dag = DAG(
    dag_id="feature_engineering_dag",
    default_args=default_args,
    description="Calculate NCAA basketball features",
    schedule_interval="0 6 * * *",  # Run daily at 6 AM
    catchup=False,
    tags=["ncaa", "basketball", "features"],
)

with dag:
    # Sensor to check if games data is available
    games_available = DuckDBTableSensor(
        task_id="games_available",
        conn_id=conn_id,
        database=database,
        table="games",
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60,  # Timeout after 1 hour
        mode="reschedule",
    )

    # Sensor to check if teams data is available
    teams_available = DuckDBTableSensor(
        task_id="teams_available",
        conn_id=conn_id,
        database=database,
        table="teams",
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60,  # Timeout after 1 hour
        mode="reschedule",
    )

    # Sensor to check if players data is available
    players_available = DuckDBTableSensor(
        task_id="players_available",
        conn_id=conn_id,
        database=database,
        table="players",
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60,  # Timeout after 1 hour
        mode="reschedule",
    )

    # Task to calculate team features
    calculate_team_features_task = PythonOperator(
        task_id="calculate_team_features",
        python_callable=calculate_team_features,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "lookback_days": lookback_days,
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
    )

    # Task to calculate player features
    calculate_player_features_task = PythonOperator(
        task_id="calculate_player_features",
        python_callable=calculate_player_features,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "lookback_days": lookback_days,
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
    )

    # Task to calculate game features
    calculate_game_features_task = PythonOperator(
        task_id="calculate_game_features",
        python_callable=calculate_game_features,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "lookback_days": lookback_days,
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
    )

    # Define task dependencies
    [games_available, teams_available, players_available] >> calculate_team_features_task
    [games_available, teams_available, players_available] >> calculate_player_features_task
    [calculate_team_features_task, calculate_player_features_task] >> calculate_game_features_task
