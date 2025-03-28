"""
Main module for the NCAA basketball prediction system.

This module provides a complete workflow for generating and evaluating predictions,
combining the various steps of the prediction process.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import polars as pl

from src.predictions.data_preparation import prepare_prediction_data
from src.predictions.prediction import (
    generate_predictions,
    format_predictions,
    calculate_prediction_accuracy,
)


def run_prediction_workflow(
    database: str,
    duckdb_conn_id: str = "duckdb_default",
    model_path: Optional[str] = None,
    model_stage: str = "production",
    model_name: str = "ncaa_basketball_prediction",
    tracking_uri: str = "sqlite:///mlflow.db",
    prediction_date: Optional[str] = None,
    lookback_days: int = 30,
    output_dir: str = "./data/predictions",
    confidence_thresholds: Optional[Dict[str, float]] = None,
    include_confidence: Optional[List[str]] = None,
    probability_format: str = "percentage",
) -> Dict[str, Any]:
    """
    Run the complete prediction workflow.

    Args:
        database: Path to the DuckDB database file
        duckdb_conn_id: Connection ID for the DuckDB connection
        model_path: Path to the model file (alternative to model_stage)
        model_stage: Model stage in registry (alternative to model_path)
        model_name: Name of the model in registry
        tracking_uri: MLflow tracking URI
        prediction_date: Date for predictions (YYYY-MM-DD format)
        lookback_days: Number of days to look back for team features
        output_dir: Directory to save output files
        confidence_thresholds: Dictionary mapping confidence levels to probability thresholds
        include_confidence: List of confidence levels to include in formatted results
        probability_format: Format for win probability

    Returns:
        Dictionary with workflow results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting NCAA basketball prediction workflow")

    # Set prediction date if not provided
    if prediction_date is None:
        prediction_date = datetime.now().strftime("%Y-%m-%d")

    # Create output directories
    data_dir = os.path.join(output_dir, "data", prediction_date)
    predictions_dir = os.path.join(output_dir, "predictions", prediction_date)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Step 1: Prepare prediction data
    logger.info("Step 1: Preparing prediction data")
    data_prep_result = prepare_prediction_data(
        conn_id=duckdb_conn_id,
        database=database,
        lookback_days=lookback_days,
        execution_date=prediction_date,
        output_path=data_dir,
    )

    if not data_prep_result["success"]:
        logger.error(f"Data preparation failed: {data_prep_result.get('error', 'Unknown error')}")
        return {
            "success": False,
            "error": f"Data preparation failed: {data_prep_result.get('error', 'Unknown error')}",
            "workflow_completed": False,
        }

    # If no games to predict, return early
    if data_prep_result["games_processed"] == 0:
        logger.info("No games to predict for the specified date")
        return {
            "success": True,
            "message": "No games to predict for the specified date",
            "workflow_completed": True,
            "games_predicted": 0,
        }

    # Step 2: Generate predictions
    logger.info("Step 2: Generating predictions")
    prediction_result = generate_predictions(
        input_path=data_dir,
        output_path=predictions_dir,
        model_path=model_path,
        model_stage=model_stage,
        model_name=model_name,
        tracking_uri=tracking_uri,
        execution_date=prediction_date,
    )

    if not prediction_result["success"]:
        logger.error(
            f"Prediction generation failed: {prediction_result.get('error', 'Unknown error')}"
        )
        return {
            "success": False,
            "error": f"Prediction generation failed: {prediction_result.get('error', 'Unknown error')}",
            "workflow_completed": False,
            "data_preparation": data_prep_result,
        }

    # Step 3: Format predictions
    logger.info("Step 3: Formatting predictions")

    # Load generated predictions
    predictions_file = os.path.join(predictions_dir, "predictions.parquet")
    predictions = pl.read_parquet(predictions_file)

    formatted_predictions = format_predictions(
        predictions=predictions,
        confidence_thresholds=confidence_thresholds,
        include_confidence=include_confidence,
        probability_format=probability_format,
        sort_by="confidence",
        sort_ascending=False,
    )

    # Save formatted predictions
    formatted_file = os.path.join(predictions_dir, "formatted_predictions.json")
    pl.DataFrame(formatted_predictions).write_json(formatted_file)

    # Return workflow results
    workflow_result = {
        "success": True,
        "workflow_completed": True,
        "data_preparation": data_prep_result,
        "prediction_generation": prediction_result,
        "games_predicted": prediction_result["games_predicted"],
        "formatted_predictions_count": len(formatted_predictions),
        "prediction_date": prediction_date,
        "output_paths": {
            "data_dir": data_dir,
            "predictions_dir": predictions_dir,
            "raw_predictions": predictions_file,
            "formatted_predictions": formatted_file,
        },
    }

    logger.info(
        f"Prediction workflow completed successfully with {prediction_result['games_predicted']} predictions"
    )
    return workflow_result


def evaluate_predictions(
    predictions_path: str, results_path: str, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate prediction accuracy by comparing with actual results.

    Args:
        predictions_path: Path to predictions file
        results_path: Path to actual results file
        output_path: Path to save evaluation results

    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating prediction accuracy")

    try:
        # Load predictions and results
        predictions = pl.read_parquet(predictions_path)
        results = pl.read_parquet(results_path)

        # Calculate accuracy
        accuracy_results = calculate_prediction_accuracy(predictions, results)

        # Save results if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Convert to DataFrame for saving
            accuracy_df = pl.DataFrame(
                {
                    "accuracy": [accuracy_results["accuracy"]],
                    "games_evaluated": [accuracy_results["games_evaluated"]],
                    "correct_predictions": [accuracy_results["correct_predictions"]],
                }
            )
            accuracy_df.write_parquet(output_path)

            logger.info(f"Evaluation results saved to {output_path}")

        return {"success": True, "evaluation_results": accuracy_results}

    except Exception as e:
        error_msg = f"Error evaluating predictions: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
