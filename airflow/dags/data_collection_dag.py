"""
DAG for collecting NCAA basketball data from the ESPN API.
This DAG fetches teams, games, players, and their statistics.
"""

from datetime import timedelta

from airflow import DAG
from airflow.operators.espn_operators import (
    FetchTeamsOperator,
    FetchGamesOperator,
    FetchPlayersOperator,
    FetchPlayerStatsOperator,
)
from airflow.sensors.data_sensors import DuckDBTableSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable


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
current_season = Variable.get("ncaa_basketball_current_season")
lookback_days = int(Variable.get("data_collection_lookback_days"))

# Database connection settings
conn_id = "duckdb_default"

# Define the DAG
dag = DAG(
    dag_id="data_collection_dag",
    default_args=default_args,
    description="Collect NCAA basketball data from ESPN API",
    schedule_interval="0 4 * * *",  # Run daily at 4 AM
    catchup=False,
    tags=["ncaa", "basketball", "data_collection"],
)

with dag:
    # Task to fetch team data
    fetch_teams = FetchTeamsOperator(
        task_id="fetch_teams",
        conn_id=conn_id,
        database=database,
        season=current_season,
        include_stats=True,
    )

    # Sensor to confirm teams data is available
    teams_available = DuckDBTableSensor(
        task_id="teams_available",
        conn_id=conn_id,
        database=database,
        table="teams",
        min_rows=300,  # Ensure we have a reasonable number of teams
        poke_interval=60,  # Check every minute
        timeout=60 * 60,  # Timeout after 1 hour
        mode="reschedule",  # Free up the worker slot while waiting
    )

    # Task to fetch game data
    fetch_games = FetchGamesOperator(
        task_id="fetch_games",
        conn_id=conn_id,
        database=database,
        lookback_days=lookback_days,  # Fetch data for the past week
        season=current_season,
        incremental_load=True,  # Only fetch new games
    )

    # Task to fetch player data
    fetch_players = FetchPlayersOperator(
        task_id="fetch_players",
        conn_id=conn_id,
        database=database,
        season=current_season,
        update_existing=True,  # Update player info if already exists
    )

    # Sensor to confirm players data is available
    players_available = DuckDBTableSensor(
        task_id="players_available",
        conn_id=conn_id,
        database=database,
        table="players",
        min_rows=3000,  # Ensure we have a reasonable number of players
        poke_interval=60,  # Check every minute
        timeout=60 * 60,  # Timeout after 1 hour
        mode="reschedule",  # Free up the worker slot while waiting
    )

    # Task to fetch player statistics
    fetch_player_stats = FetchPlayerStatsOperator(
        task_id="fetch_player_stats",
        conn_id=conn_id,
        database=database,
        lookback_days=lookback_days,  # Fetch stats for the past week
        season=current_season,
    )

    # Define task dependencies
    fetch_teams >> teams_available >> [fetch_games, fetch_players]
    fetch_players >> players_available >> fetch_player_stats
