"""
Airflow DAG for NCAA basketball prediction evaluation workflow.

This DAG orchestrates the evaluation of previous predictions against actual game results:
1. Fetch completed games for a date range
2. Load predictions for those games
3. Compare predictions to actual results and calculate accuracy
4. Store evaluation metrics
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models.param import Param

# Import our evaluation functions
from src.predictions.main import evaluate_predictions


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
    "ncaa_basketball_prediction_evaluation",
    default_args=default_args,
    description="NCAA Basketball prediction evaluation workflow",
    schedule_interval="0 10 * * *",  # Run daily at 10 AM
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ncaa", "basketball", "prediction", "evaluation"],
    params={
        "duckdb_conn_id": Param("duckdb_default", type="string"),
        "database_path": Param("data/basketball.db", type="string"),
        "predictions_dir": Param("data/predictions", type="string"),
        "evaluation_dir": Param("data/evaluations", type="string"),
        "days_to_evaluate": Param(1, type="integer"),
    },
)


# Task to evaluate predictions against actual results
def evaluate_predictions_task(**kwargs):
    """Task to evaluate predictions against actual game results."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    conn_id = kwargs["params"]["duckdb_conn_id"]
    database = kwargs["params"]["database_path"]
    predictions_dir = kwargs["params"]["predictions_dir"]
    evaluation_dir = kwargs["params"]["evaluation_dir"]
    days_to_evaluate = kwargs["params"]["days_to_evaluate"]

    # Calculate the date to evaluate (usually yesterday)
    # We evaluate predictions from N days ago based on the parameter
    eval_date = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=days_to_evaluate)
    eval_date_str = eval_date.strftime("%Y-%m-%d")

    # Path to the predictions file
    prediction_file = os.path.join(predictions_dir, f"predictions_{eval_date_str}.json")

    # Output path for evaluation results
    output_path = os.path.join(evaluation_dir, f"evaluation_{eval_date_str}.json")

    # Evaluate predictions
    result = evaluate_predictions(
        conn_id=conn_id,
        database=database,
        prediction_file=prediction_file,
        output_path=output_path,
        evaluation_date=eval_date_str,
    )

    # Push the result to XCom
    ti.xcom_push(key="evaluation_result", value=result)

    # Return success status for Airflow
    if result.get("success", False):
        return (
            f"Successfully evaluated {result.get('games_evaluated', 0)} predictions "
            f"with accuracy: {result.get('accuracy', 0):.4f}"
        )
    else:
        raise Exception(f"Failed to evaluate predictions: {result.get('error', 'Unknown error')}")


# Define the tasks
evaluate_predictions_task = PythonOperator(
    task_id="evaluate_predictions",
    python_callable=evaluate_predictions_task,
    provide_context=True,
    dag=dag,
)

# Simple workflow with just one primary task
workflow_start = DummyOperator(
    task_id="workflow_start",
    dag=dag,
)

workflow_end = DummyOperator(
    task_id="workflow_end",
    dag=dag,
)

# Set task dependencies
workflow_start >> evaluate_predictions_task >> workflow_end
