"""
Airflow DAG for NCAA basketball prediction workflow.

This DAG orchestrates the end-to-end workflow for generating predictions
for upcoming NCAA basketball games:
1. Fetch upcoming games from ESPN API
2. Prepare feature data for predictions
3. Generate predictions using the registered model
4. Format and store prediction results
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

# Import our workflow functions
from src.predictions.upcoming_games import fetch_upcoming_games
from src.predictions.main import run_prediction_workflow


# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "ncaa_basketball_prediction",
    default_args=default_args,
    description="NCAA Basketball prediction workflow",
    schedule_interval="0 6 * * *",  # Run every day at 6 AM
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ncaa", "basketball", "prediction"],
    params={
        "duckdb_conn_id": Param("duckdb_default", type="string"),
        "database_path": Param("data/basketball.db", type="string"),
        "days_ahead": Param(7, type="integer"),
        "lookback_days": Param(30, type="integer"),
        "model_stage": Param("production", type="string"),
        "model_name": Param("ncaa_basketball_prediction", type="string"),
        "tracking_uri": Param("sqlite:///mlflow.db", type="string"),
        "output_dir": Param("data/predictions", type="string"),
    },
)


# Task to fetch upcoming games
def fetch_games_task(**kwargs):
    """Task to fetch upcoming games from ESPN API."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    conn_id = kwargs["params"]["duckdb_conn_id"]
    database = kwargs["params"]["database_path"]
    days_ahead = kwargs["params"]["days_ahead"]

    # Fetch upcoming games
    result = fetch_upcoming_games(
        conn_id=conn_id, database=database, days_ahead=days_ahead, execution_date=execution_date
    )

    # Push the result to XCom
    ti.xcom_push(key="fetch_games_result", value=result)

    # Return success status for Airflow
    if result["success"]:
        return f"Successfully fetched {result['new_games']} new games"
    else:
        raise Exception(f"Failed to fetch games: {result.get('error', 'Unknown error')}")


# Task to run the complete prediction workflow
def prediction_workflow_task(**kwargs):
    """Task to run the complete prediction workflow."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    conn_id = kwargs["params"]["duckdb_conn_id"]
    database = kwargs["params"]["database_path"]
    lookback_days = kwargs["params"]["lookback_days"]
    model_stage = kwargs["params"]["model_stage"]
    model_name = kwargs["params"]["model_name"]
    tracking_uri = kwargs["params"]["tracking_uri"]
    output_dir = kwargs["params"]["output_dir"]

    # Get fetch games result from XCom
    fetch_result = ti.xcom_pull(task_ids="fetch_upcoming_games", key="fetch_games_result")

    # Only run prediction if we have new games
    if fetch_result and fetch_result.get("new_games", 0) > 0:
        # Run prediction workflow
        result = run_prediction_workflow(
            database=database,
            duckdb_conn_id=conn_id,
            model_stage=model_stage,
            model_name=model_name,
            tracking_uri=tracking_uri,
            prediction_date=execution_date,
            lookback_days=lookback_days,
            output_dir=output_dir,
        )

        # Push the result to XCom
        ti.xcom_push(key="prediction_result", value=result)

        # Return success status for Airflow
        if result["success"]:
            return (
                f"Successfully generated predictions for {result.get('games_predicted', 0)} games"
            )
        else:
            raise Exception(
                f"Failed to generate predictions: {result.get('error', 'Unknown error')}"
            )
    else:
        return "No new games to predict"


# Define the tasks
fetch_upcoming_games_task = PythonOperator(
    task_id="fetch_upcoming_games",
    python_callable=fetch_games_task,
    provide_context=True,
    dag=dag,
)

generate_predictions_task = PythonOperator(
    task_id="generate_predictions",
    python_callable=prediction_workflow_task,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
fetch_upcoming_games_task >> generate_predictions_task
