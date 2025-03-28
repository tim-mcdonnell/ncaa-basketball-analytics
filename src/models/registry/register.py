"""
Module for registering models in the MLflow model registry.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from mlflow.tracking import MlflowClient

from src.models.evaluation import evaluate_model


def register_model(
    model_path: str,
    model_name: str,
    tracking_uri: str,
    min_accuracy: float = 0.7,
    execution_date: Optional[str] = None,
    test_data_path: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
    stage: str = "None",
) -> Dict[str, Any]:
    """
    Register a trained model in the MLflow model registry if it meets quality criteria.

    This function:
    1. Loads model metadata
    2. Evaluates model quality (either using provided metrics or by evaluating on test data)
    3. Checks if model meets quality thresholds
    4. Registers the model in the MLflow model registry if quality criteria are met

    Args:
        model_path: Path to the directory containing the trained model
        model_name: Name to use in the model registry
        tracking_uri: MLflow tracking URI
        min_accuracy: Minimum accuracy threshold for registration (default: 0.7)
        execution_date: Execution date in YYYY-MM-DD format (defaults to today)
        test_data_path: Path to test data for evaluation (required if metrics not provided)
        metrics: Pre-computed metrics for the model (if not provided, will evaluate using test data)
        stage: Initial stage for the model (default: None)

    Returns:
        Dictionary with registration results
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Set execution date to today if not provided
    if execution_date is None:
        execution_date = datetime.now().strftime("%Y-%m-%d")

    try:
        # Verify model files exist
        if not os.path.exists(model_path):
            return {
                "success": False,
                "error": f"Model path not found: {model_path}",
                "execution_date": execution_date,
            }

        config_file = os.path.join(model_path, "model_config.json")
        model_file = os.path.join(model_path, "model.pt")

        if not os.path.exists(config_file):
            return {
                "success": False,
                "error": f"Model config file not found: {config_file}",
                "execution_date": execution_date,
            }

        if not os.path.exists(model_file):
            return {
                "success": False,
                "error": f"Model file not found: {model_file}",
                "execution_date": execution_date,
            }

        # Load model config
        with open(config_file, "r") as f:
            model_config = json.load(f)

        # Get model metrics
        if metrics is None:
            # If no metrics provided, evaluate the model
            if test_data_path is None:
                return {
                    "success": False,
                    "error": "Either metrics or test_data_path must be provided",
                    "execution_date": execution_date,
                }

            logger.info(f"Evaluating model on test data: {test_data_path}")
            eval_result = evaluate_model(
                model_path=model_path,
                test_data_path=test_data_path,
                tracking_uri=tracking_uri,
                execution_date=execution_date,
            )

            if not eval_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to evaluate model: {eval_result.get('error', 'Unknown error')}",
                    "execution_date": execution_date,
                }

            metrics = eval_result.get("metrics", {})

        # Extract accuracy
        accuracy = metrics.get("accuracy", 0.0)

        # Check if model meets quality criteria
        if accuracy < min_accuracy:
            logger.info(
                f"Model accuracy {accuracy:.4f} below threshold {min_accuracy:.4f}, not registering"
            )
            return {
                "success": True,
                "registered": False,
                "accuracy_threshold": min_accuracy,
                "actual_accuracy": accuracy,
                "execution_date": execution_date,
            }

        # Register model in MLflow
        logger.info(f"Registering model {model_name} in MLflow registry")
        client = MlflowClient(tracking_uri=tracking_uri)

        # Check if registered model exists, create if not
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)

        # Create model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_path,
            run_id=None,  # No run ID needed for direct registration
        )

        # Set model version stage if specified
        if stage != "None":
            client.transition_model_version_stage(
                name=model_name, version=model_version.version, stage=stage
            )

        # Add model metadata as tags
        try:
            for key, value in model_config.items():
                if isinstance(value, (str, int, float, bool)):
                    client.set_model_version_tag(
                        name=model_name, version=model_version.version, key=key, value=str(value)
                    )

            # Add metrics as tags
            for key, value in metrics.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=f"metric.{key}",
                    value=str(value),
                )

            # Add registration date
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="registration_date",
                value=execution_date,
            )
        except Exception as e:
            logger.warning(f"Failed to set some model tags: {str(e)}")

        logger.info(
            f"Model registered successfully as {model_name} version {model_version.version}"
        )

        return {
            "success": True,
            "registered": True,
            "model_name": model_name,
            "model_version": model_version.version,
            "accuracy": accuracy,
            "stage": stage,
            "execution_date": execution_date,
        }

    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        return {"success": False, "error": str(e), "execution_date": execution_date}
