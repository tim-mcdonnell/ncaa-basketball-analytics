"""
DAG for calculating NCAA basketball features from collected data.
This DAG calculates team, player, and game features for model training.
"""

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.data_sensors import NewDataSensor
from airflow.utils.dates import days_ago

# Import feature calculation functions from project
# These would be implemented in the src/features directory
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

# Database connection settings
conn_id = "duckdb_default"
database = "/path/to/ncaa_basketball.duckdb"  # Update path as needed

# Define the DAG
dag = DAG(
    dag_id="feature_engineering_dag",
    default_args=default_args,
    description="Calculate NCAA basketball features for model training",
    schedule_interval="0 6 * * *",  # Run daily at 6 AM (after data collection)
    catchup=False,
    tags=["ncaa", "basketball", "feature_engineering"],
)

with dag:
    # Sensor to check if new game data is available
    new_games_available = NewDataSensor(
        task_id="new_games_available",
        conn_id=conn_id,
        database=database,
        table="games",
        date_column="date",
        execution_date="{{ execution_date.strftime('%Y-%m-%d') }}",
        mode="reschedule",
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60 * 2,  # Timeout after 2 hours
    )

    # Sensor to check if new player stats are available
    new_player_stats_available = NewDataSensor(
        task_id="new_player_stats_available",
        conn_id=conn_id,
        database=database,
        sql="""
        SELECT COUNT(*) FROM player_stats ps
        JOIN games g ON ps.game_id = g.game_id
        WHERE g.date > '{{ execution_date.strftime('%Y-%m-%d') }}'
        """,
        mode="reschedule",
        poke_interval=300,  # Check every 5 minutes
        timeout=60 * 60 * 2,  # Timeout after 2 hours
    )

    # Task to calculate team features
    calculate_team_features_task = PythonOperator(
        task_id="calculate_team_features",
        python_callable=calculate_team_features,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "lookback_days": 30,  # Calculate features using data from last 30 days
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=2,
    )

    # Task to calculate player features
    calculate_player_features_task = PythonOperator(
        task_id="calculate_player_features",
        python_callable=calculate_player_features,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "lookback_days": 30,  # Calculate features using data from last 30 days
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=2,
    )

    # Task to calculate game features
    calculate_game_features_task = PythonOperator(
        task_id="calculate_game_features",
        python_callable=calculate_game_features,
        op_kwargs={
            "conn_id": conn_id,
            "database": database,
            "lookback_days": 30,  # Calculate features using data from last 30 days
            "execution_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        },
        retries=2,
    )

    # Define task dependencies
    [new_games_available, new_player_stats_available] >> [
        calculate_team_features_task,
        calculate_player_features_task,
    ]

    [calculate_team_features_task, calculate_player_features_task] >> calculate_game_features_task
