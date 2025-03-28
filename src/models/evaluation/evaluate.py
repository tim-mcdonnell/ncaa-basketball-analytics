"""
Module for evaluating trained models.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import polars as pl
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error

from src.models.mlflow.tracking import MLflowTracker
from src.models.inference.predictor import batch_predict


def evaluate_model(
    model_path: str,
    test_data_path: str,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "ncaa_basketball_predictions",
    execution_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    This function:
    1. Loads a trained model and test data
    2. Makes predictions on the test data
    3. Calculates evaluation metrics
    4. Logs metrics to MLflow
    5. Returns a summary of the evaluation

    Args:
        model_path: Path to the directory containing the trained model
        test_data_path: Path to the test data parquet file
        tracking_uri: MLflow tracking URI (default: None)
        experiment_name: Name of the MLflow experiment (default: ncaa_basketball_predictions)
        execution_date: Execution date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dictionary with evaluation results
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Set execution date to today if not provided
    if execution_date is None:
        execution_date = datetime.now().strftime("%Y-%m-%d")

    try:
        # Load the model
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)

        # Get feature columns from model config
        feature_columns = model.config.features
        model_type = model.config.model_type

        # Load test data
        logger.info(f"Loading test data from {test_data_path}")
        test_data = pl.read_parquet(test_data_path)

        # Initialize MLflow tracker if tracking URI is provided
        if tracking_uri:
            mlflow_tracker = MLflowTracker(tracking_uri=tracking_uri)
            mlflow_tracker.set_experiment(experiment_name)
            run_id = mlflow_tracker.start_run(run_name=f"evaluate_{model_type}_{execution_date}")

            # Log model parameters
            mlflow_tracker.log_params(
                {
                    "model_type": model_type,
                    "execution_date": execution_date,
                    "test_samples": len(test_data),
                    "evaluation_type": "test_set",
                }
            )
        else:
            mlflow_tracker = None
            run_id = None

        # Make predictions
        logger.info("Making predictions on test data")
        predictions = batch_predict(model, test_data, feature_columns)

        # Get true values
        y_true_win = test_data["home_win"].to_numpy()
        y_true_point_diff = test_data["point_diff"].to_numpy()

        # Convert predictions to numpy
        y_pred_win_prob = predictions.detach().numpy().flatten()
        # Binary predictions not used in this function, will be calculated in calculate_metrics

        # Calculate metrics
        logger.info("Calculating evaluation metrics")
        metrics = calculate_metrics(y_true_win, y_pred_win_prob, y_true_point_diff)

        # Log metrics to MLflow
        if mlflow_tracker:
            mlflow_tracker.log_metrics(metrics)
            mlflow_tracker.end_run()

        logger.info(
            f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}"
        )

        # Return results
        return {
            "success": True,
            "model_type": model_type,
            "metrics": metrics,
            "test_samples": len(test_data),
            "run_id": run_id,
            "execution_date": execution_date,
        }

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return {"success": False, "error": str(e), "execution_date": execution_date}


def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the directory containing the model

    Returns:
        Loaded model instance

    Raises:
        FileNotFoundError: If model files are not found
    """
    # Check if model files exist
    model_file = os.path.join(model_path, "model.pt")
    config_file = os.path.join(model_path, "model_config.json")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Model config file not found: {config_file}")

    # Load model config
    with open(config_file, "r") as f:
        config_data = json.load(f)

    # Determine model type from config
    model_type = config_data.get("model_type", "gradient_boosting")

    # Load the correct model based on type
    if model_type == "gradient_boosting":
        from src.models.game_prediction.basic_model import BasicGamePredictionModel

        model = BasicGamePredictionModel.load(model_file)
    elif model_type == "neural_network":
        from src.models.game_prediction.neural_model import NeuralGamePredictionModel

        model = NeuralGamePredictionModel.load(model_file)
    elif model_type == "logistic_regression":
        from src.models.game_prediction.basic_model import LogisticRegressionModel

        model = LogisticRegressionModel.load(model_file)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def calculate_metrics(
    y_true_win: np.ndarray, y_pred_win_prob: np.ndarray, y_true_point_diff: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the model predictions.

    Args:
        y_true_win: True binary win labels
        y_pred_win_prob: Predicted win probabilities
        y_true_point_diff: True point differentials

    Returns:
        Dictionary of metric names and values
    """
    # Convert predictions to binary if needed
    y_pred_win = y_pred_win_prob >= 0.5

    # Classification metrics
    accuracy = accuracy_score(y_true_win, y_pred_win)
    auc = roc_auc_score(y_true_win, y_pred_win_prob)

    # Regression metrics (for point differential)
    # Create a simple point diff prediction from win probability
    # This is a placeholder - in reality, we'd use a proper regression model
    y_pred_point_diff = (y_pred_win_prob - 0.5) * 20  # Simple scaling

    mse = mean_squared_error(y_true_point_diff, y_pred_point_diff)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_point_diff, y_pred_point_diff)

    return {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "correct_predictions": int(np.sum(y_pred_win == y_true_win)),
        "total_predictions": len(y_true_win),
    }
