"""
Airflow DAG for NCAA basketball model training workflow.

This DAG orchestrates the end-to-end workflow for training, evaluating, and registering
NCAA basketball prediction models:
1. Prepare training data
2. Train prediction model
3. Evaluate model performance
4. Register model in MLflow if it meets quality criteria
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models.param import Param
from airflow.utils.trigger_rule import TriggerRule

# Import our workflow functions
from src.models.data_preparation import prepare_training_data
from src.models.training import train_model
from src.models.evaluation import evaluate_model
from src.models.registry import register_model


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
    "ncaa_basketball_model_training",
    default_args=default_args,
    description="NCAA Basketball model training workflow",
    schedule_interval="0 0 * * 0",  # Run every Sunday at midnight
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ncaa", "basketball", "model", "training"],
    params={
        "duckdb_conn_id": Param("duckdb_default", type="string"),
        "database_path": Param("data/basketball.db", type="string"),
        "lookback_days": Param(365, type="integer"),
        "model_type": Param(
            "gradient_boosting",
            type="string",
            enum=["gradient_boosting", "neural_network", "logistic_regression"],
        ),
        "tracking_uri": Param("sqlite:///mlflow.db", type="string"),
        "model_name": Param("ncaa_basketball_prediction", type="string"),
        "min_accuracy": Param(0.7, type="number"),
        "output_dir": Param("data/models", type="string"),
    },
)


# Task to prepare training data
def prepare_data_task(**kwargs):
    """Task to prepare training data for model training."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    conn_id = kwargs["params"]["duckdb_conn_id"]
    database = kwargs["params"]["database_path"]
    lookback_days = kwargs["params"]["lookback_days"]
    output_dir = os.path.join(kwargs["params"]["output_dir"], execution_date, "training_data")

    # Prepare training data
    result = prepare_training_data(
        conn_id=conn_id,
        database=database,
        lookback_days=lookback_days,
        execution_date=execution_date,
        output_path=output_dir,
    )

    # Push the result to XCom
    ti.xcom_push(key="prepare_data_result", value=result)

    # Return success status for Airflow
    if result.get("success", False):
        return f"Successfully prepared training data with {result.get('games_processed', 0)} games"
    else:
        raise Exception(f"Failed to prepare training data: {result.get('error', 'Unknown error')}")


# Task to train the model
def train_model_task(**kwargs):
    """Task to train the prediction model."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    model_type = kwargs["params"]["model_type"]
    tracking_uri = kwargs["params"]["tracking_uri"]
    output_base_dir = kwargs["params"]["output_dir"]

    # Define paths - no need to use data_result variable
    input_path = os.path.join(output_base_dir, execution_date, "training_data")
    output_path = os.path.join(output_base_dir, execution_date, "trained_model")

    # Train model
    result = train_model(
        input_path=input_path,
        output_path=output_path,
        model_type=model_type,
        tracking_uri=tracking_uri,
        execution_date=execution_date,
    )

    # Push the result to XCom
    ti.xcom_push(key="train_model_result", value=result)

    # Return success status for Airflow
    if result.get("success", False):
        return f"Successfully trained {model_type} model"
    else:
        raise Exception(f"Failed to train model: {result.get('error', 'Unknown error')}")


# Task to evaluate the model
def evaluate_model_task(**kwargs):
    """Task to evaluate the trained model performance."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    tracking_uri = kwargs["params"]["tracking_uri"]
    output_base_dir = kwargs["params"]["output_dir"]

    # Define paths - no need to use model_result variable
    model_path = os.path.join(output_base_dir, execution_date, "trained_model")
    test_data_path = os.path.join(
        output_base_dir, execution_date, "training_data", "test_data.parquet"
    )

    # Evaluate model
    result = evaluate_model(
        model_path=model_path,
        test_data_path=test_data_path,
        tracking_uri=tracking_uri,
        execution_date=execution_date,
    )

    # Push the result to XCom
    ti.xcom_push(key="evaluate_model_result", value=result)

    # Return success status for Airflow
    if result.get("success", False):
        return f"Model evaluated successfully. Accuracy: {result.get('metrics', {}).get('accuracy', 0):.4f}"
    else:
        raise Exception(f"Failed to evaluate model: {result.get('error', 'Unknown error')}")


# Task to decide whether to register the model
def decide_registration_task(**kwargs):
    """Task to decide whether to register the model based on accuracy thresholds."""
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    min_accuracy = float(kwargs["params"]["min_accuracy"])

    # Get evaluation result
    eval_result = ti.xcom_pull(task_ids="evaluate_model", key="evaluate_model_result")

    # Check if model meets accuracy threshold
    if eval_result and eval_result.get("success", False):
        accuracy = eval_result.get("metrics", {}).get("accuracy", 0)
        if accuracy >= min_accuracy:
            return "register_model"
        else:
            return "skip_registration"
    else:
        return "skip_registration"


# Task to register the model if quality criteria are met
def register_model_task(**kwargs):
    """Task to register the model in MLflow if it meets quality criteria."""
    execution_date = kwargs["ds"]  # Airflow provides the execution date
    ti = kwargs["ti"]  # Task instance

    # Get parameters
    tracking_uri = kwargs["params"]["tracking_uri"]
    model_name = kwargs["params"]["model_name"]
    min_accuracy = float(kwargs["params"]["min_accuracy"])
    output_base_dir = kwargs["params"]["output_dir"]

    # Get evaluation result
    eval_result = ti.xcom_pull(task_ids="evaluate_model", key="evaluate_model_result")
    model_path = os.path.join(output_base_dir, execution_date, "trained_model")

    # Register model
    result = register_model(
        model_path=model_path,
        model_name=model_name,
        tracking_uri=tracking_uri,
        min_accuracy=min_accuracy,
        metrics=eval_result.get("metrics", {}),
        execution_date=execution_date,
        stage="Production"
        if eval_result.get("metrics", {}).get("accuracy", 0) > 0.8
        else "Staging",
    )

    # Push the result to XCom
    ti.xcom_push(key="register_model_result", value=result)

    # Return success status for Airflow
    if result.get("success", False):
        if result.get("registered", False):
            return f"Model registered successfully as {result.get('model_name')} v{result.get('model_version')}"
        else:
            return f"Model not registered. Accuracy {result.get('actual_accuracy', 0):.4f} below threshold {result.get('accuracy_threshold', 0):.4f}"
    else:
        raise Exception(f"Failed to register model: {result.get('error', 'Unknown error')}")


# Define the tasks
prepare_training_data_task = PythonOperator(
    task_id="prepare_training_data",
    python_callable=prepare_data_task,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model_task,
    provide_context=True,
    dag=dag,
)

decide_registration_task = BranchPythonOperator(
    task_id="decide_registration",
    python_callable=decide_registration_task,
    provide_context=True,
    dag=dag,
)

register_model_task = PythonOperator(
    task_id="register_model",
    python_callable=register_model_task,
    provide_context=True,
    dag=dag,
)

skip_registration_task = DummyOperator(
    task_id="skip_registration",
    dag=dag,
)

workflow_end = DummyOperator(
    task_id="workflow_end",
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)

# Set task dependencies
prepare_training_data_task >> train_model_task >> evaluate_model_task >> decide_registration_task
decide_registration_task >> [register_model_task, skip_registration_task]
[register_model_task, skip_registration_task] >> workflow_end
