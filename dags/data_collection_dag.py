"""
Airflow DAG for NCAA basketball data collection and ingestion.

This DAG orchestrates the collection and ingestion of NCAA basketball game data:
1. Fetch completed games data from various sources
2. Process and clean the raw game data
3. Store the data in the database for future use in training and prediction
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models.param import Param

# Import our data collection functions
# Note: These functions would need to be implemented in the appropriate modules
from src.data.collection import fetch_game_results, fetch_team_stats
from src.data.ingestion import process_game_data, process_team_stats


# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "ncaa_basketball_data_collection",
    default_args=default_args,
    description="NCAA Basketball data collection and ingestion workflow",
    schedule_interval="0 4 * * *",  # Run every day at 4 AM
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ncaa", "basketball", "data", "collection"],
    params={
        "duckdb_conn_id": Param("duckdb_default", type="string"),
        "database_path": Param("data/basketball.db", type="string"),
        "raw_data_dir": Param("data/raw", type="string"),
        "processed_data_dir": Param("data/processed", type="string"),
        "days_to_fetch": Param(1, type="integer"),
    },
)


# Task to fetch game results
def fetch_game_results_task(**kwargs):
    """Task to fetch NCAA basketball game results."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    raw_data_dir = kwargs["params"]["raw_data_dir"]
    days_to_fetch = kwargs["params"]["days_to_fetch"]

    # Calculate start date (usually yesterday)
    start_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=days_to_fetch)
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Path for storing raw game data
    output_path = os.path.join(raw_data_dir, f"game_results_{start_date_str}.json")

    # Fetch game results
    result = fetch_game_results(
        start_date=start_date_str, end_date=execution_date, output_path=output_path
    )

    # Push the result to XCom
    ti.xcom_push(key="game_results", value=result)

    # Return success status for Airflow
    if result.get("success", False):
        return f"Successfully fetched {result.get('games_fetched', 0)} game results"
    else:
        raise Exception(f"Failed to fetch game results: {result.get('error', 'Unknown error')}")


# Task to fetch team statistics
def fetch_team_stats_task(**kwargs):
    """Task to fetch NCAA basketball team statistics."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    raw_data_dir = kwargs["params"]["raw_data_dir"]
    days_to_fetch = kwargs["params"]["days_to_fetch"]

    # Calculate start date (usually yesterday)
    start_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=days_to_fetch)
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Path for storing raw team stats
    output_path = os.path.join(raw_data_dir, f"team_stats_{start_date_str}.json")

    # Fetch team statistics
    result = fetch_team_stats(
        start_date=start_date_str, end_date=execution_date, output_path=output_path
    )

    # Push the result to XCom
    ti.xcom_push(key="team_stats", value=result)

    # Return success status for Airflow
    if result.get("success", False):
        return f"Successfully fetched statistics for {result.get('teams_fetched', 0)} teams"
    else:
        raise Exception(f"Failed to fetch team statistics: {result.get('error', 'Unknown error')}")


# Task to process and ingest game data
def process_game_data_task(**kwargs):
    """Task to process and ingest game data into the database."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    conn_id = kwargs["params"]["duckdb_conn_id"]
    database = kwargs["params"]["database_path"]
    raw_data_dir = kwargs["params"]["raw_data_dir"]
    processed_data_dir = kwargs["params"]["processed_data_dir"]
    days_to_fetch = kwargs["params"]["days_to_fetch"]

    # Calculate date for data to process
    data_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=days_to_fetch)
    data_date_str = data_date.strftime("%Y-%m-%d")

    # Paths for input and output data
    input_path = os.path.join(raw_data_dir, f"game_results_{data_date_str}.json")
    output_path = os.path.join(processed_data_dir, f"processed_games_{data_date_str}.parquet")

    # Get fetch result from XCom to check if there's data to process
    fetch_result = ti.xcom_pull(task_ids="fetch_game_results", key="game_results")

    if fetch_result and fetch_result.get("games_fetched", 0) > 0:
        # Process and ingest game data
        result = process_game_data(
            input_path=input_path, output_path=output_path, conn_id=conn_id, database=database
        )

        # Push the result to XCom
        ti.xcom_push(key="process_games", value=result)

        # Return success status for Airflow
        if result.get("success", False):
            return f"Successfully processed and ingested {result.get('games_processed', 0)} games"
        else:
            raise Exception(f"Failed to process game data: {result.get('error', 'Unknown error')}")
    else:
        return "No game results to process"


# Task to process and ingest team statistics
def process_team_stats_task(**kwargs):
    """Task to process and ingest team statistics into the database."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    conn_id = kwargs["params"]["duckdb_conn_id"]
    database = kwargs["params"]["database_path"]
    raw_data_dir = kwargs["params"]["raw_data_dir"]
    processed_data_dir = kwargs["params"]["processed_data_dir"]
    days_to_fetch = kwargs["params"]["days_to_fetch"]

    # Calculate date for data to process
    data_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=days_to_fetch)
    data_date_str = data_date.strftime("%Y-%m-%d")

    # Paths for input and output data
    input_path = os.path.join(raw_data_dir, f"team_stats_{data_date_str}.json")
    output_path = os.path.join(processed_data_dir, f"processed_team_stats_{data_date_str}.parquet")

    # Get fetch result from XCom to check if there's data to process
    fetch_result = ti.xcom_pull(task_ids="fetch_team_stats", key="team_stats")

    if fetch_result and fetch_result.get("teams_fetched", 0) > 0:
        # Process and ingest team statistics
        result = process_team_stats(
            input_path=input_path, output_path=output_path, conn_id=conn_id, database=database
        )

        # Push the result to XCom
        ti.xcom_push(key="process_stats", value=result)

        # Return success status for Airflow
        if result.get("success", False):
            return f"Successfully processed and ingested statistics for {result.get('teams_processed', 0)} teams"
        else:
            raise Exception(
                f"Failed to process team statistics: {result.get('error', 'Unknown error')}"
            )
    else:
        return "No team statistics to process"


# Define the tasks
fetch_game_results_task = PythonOperator(
    task_id="fetch_game_results",
    python_callable=fetch_game_results_task,
    provide_context=True,
    dag=dag,
)

fetch_team_stats_task = PythonOperator(
    task_id="fetch_team_stats",
    python_callable=fetch_team_stats_task,
    provide_context=True,
    dag=dag,
)

process_game_data_task = PythonOperator(
    task_id="process_game_data",
    python_callable=process_game_data_task,
    provide_context=True,
    dag=dag,
)

process_team_stats_task = PythonOperator(
    task_id="process_team_stats",
    python_callable=process_team_stats_task,
    provide_context=True,
    dag=dag,
)

workflow_start = DummyOperator(
    task_id="workflow_start",
    dag=dag,
)

workflow_end = DummyOperator(
    task_id="workflow_end",
    dag=dag,
)

# Set task dependencies
workflow_start >> [fetch_game_results_task, fetch_team_stats_task]
fetch_game_results_task >> process_game_data_task
fetch_team_stats_task >> process_team_stats_task
[process_game_data_task, process_team_stats_task] >> workflow_end
