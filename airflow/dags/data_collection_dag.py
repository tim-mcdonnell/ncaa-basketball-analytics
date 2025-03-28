"""
DAG for collecting NCAA basketball data from ESPN API.
This DAG fetches team, game, player, and player stats data.
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
current_season = "2023-24"  # Update as needed

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
        retries=5,  # Override default retries for this critical task
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
        start_date="{{ execution_date.strftime('%Y-%m-%d') }}",
        end_date="{{ (execution_date + macros.timedelta(days=7)).strftime('%Y-%m-%d') }}",
        incremental=True,
        retries=5,  # Override default retries for this critical task
    )

    # Task to fetch player data
    fetch_players = FetchPlayersOperator(
        task_id="fetch_players",
        conn_id=conn_id,
        database=database,
        season=current_season,
        retries=5,  # Override default retries for this critical task
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

    # Task to fetch player stats for recent and upcoming games
    fetch_player_stats = FetchPlayerStatsOperator(
        task_id="fetch_player_stats",
        conn_id=conn_id,
        database=database,
        start_date="{{ (execution_date - macros.timedelta(days=7)).strftime('%Y-%m-%d') }}",
        end_date="{{ execution_date.strftime('%Y-%m-%d') }}",
        retries=5,  # Override default retries for this critical task
    )

    # Define task dependencies
    fetch_teams >> teams_available >> [fetch_games, fetch_players]
    fetch_players >> players_available >> fetch_player_stats
