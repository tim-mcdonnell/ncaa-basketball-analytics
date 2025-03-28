"""
Module for training models.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import polars as pl

from src.models.base import ModelConfig, ModelVersion
from src.models.training.trainer import ModelTrainer
from src.models.mlflow.tracking import MLflowTracker


def train_model(
    input_path: str,
    output_path: str,
    model_type: str = "gradient_boosting",
    hyperparameters: Optional[Dict[str, Any]] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "ncaa_basketball_predictions",
    execution_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a model using prepared training data.

    This function:
    1. Loads training and validation data
    2. Creates a model of the specified type
    3. Trains the model using the training data
    4. Evaluates the model using validation data
    5. Saves the model and its configuration
    6. Logs training metrics and artifacts to MLflow

    Args:
        input_path: Path to the directory containing training data
        output_path: Path to save the trained model
        model_type: Type of model to train (default: gradient_boosting)
        hyperparameters: Model-specific hyperparameters (default: None)
        tracking_uri: MLflow tracking URI (default: None)
        experiment_name: Name of the MLflow experiment (default: ncaa_basketball_predictions)
        execution_date: Execution date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dictionary with training results
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Set execution date to today if not provided
    if execution_date is None:
        execution_date = datetime.now().strftime("%Y-%m-%d")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    try:
        # Load training and validation data
        logger.info(f"Loading training data from {input_path}")
        train_data, val_data, feature_columns = load_training_data(input_path)

        # Initialize MLflow tracker if tracking URI is provided
        if tracking_uri:
            mlflow_tracker = MLflowTracker(tracking_uri=tracking_uri)
            mlflow_tracker.set_experiment(experiment_name)
            run_id = mlflow_tracker.start_run(run_name=f"train_{model_type}_{execution_date}")
        else:
            mlflow_tracker = None
            run_id = None

        # Set default hyperparameters if not provided
        if hyperparameters is None:
            hyperparameters = get_default_hyperparameters(model_type)

        # Log training parameters to MLflow
        if mlflow_tracker:
            mlflow_tracker.log_params(
                {
                    "model_type": model_type,
                    "execution_date": execution_date,
                    "feature_count": len(feature_columns),
                    "train_samples": len(train_data),
                    "val_samples": len(val_data),
                    **hyperparameters,
                }
            )

        # Create model
        logger.info(f"Creating {model_type} model")
        model = create_model(
            model_type=model_type, input_features=feature_columns, hyperparameters=hyperparameters
        )

        # Create model configuration
        model_config = ModelConfig(
            model_type=model_type,
            hyperparameters=hyperparameters,
            features=feature_columns,
            training_params={
                "training_date": execution_date,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
            },
            version=ModelVersion.from_date(execution_date),
        )

        # Prepare data for training
        logger.info("Preparing data for training")
        train_features = train_data.select(feature_columns)
        train_targets = train_data.select(["home_win", "point_diff"])

        val_features = val_data.select(feature_columns)
        val_targets = val_data.select(["home_win", "point_diff"])

        # Initialize model trainer
        trainer = ModelTrainer(
            model=model,
            train_data=(train_features, train_targets),
            val_data=(val_features, val_targets),
            config=model_config,
        )

        # Train the model
        logger.info("Training model")
        training_history = trainer.train()

        # Extract final metrics
        metrics = {
            "train_loss": training_history["train_loss"][-1],
            "val_loss": training_history["val_loss"][-1],
            "training_epochs": len(training_history["train_loss"]),
        }

        # Log metrics to MLflow
        if mlflow_tracker:
            mlflow_tracker.log_metrics(metrics)

        # Save the model
        model_path = os.path.join(output_path, "model.pt")
        logger.info(f"Saving model to {model_path}")
        model.save(model_path)

        # Save model configuration
        config_path = os.path.join(output_path, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(model_config.dict(), f, indent=2)

        # Log artifacts to MLflow
        if mlflow_tracker:
            mlflow_tracker.log_artifacts(output_path)
            mlflow_tracker.end_run()

        # Return results
        return {
            "success": True,
            "model_type": model_type,
            "model_path": model_path,
            "config_path": config_path,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "run_id": run_id,
            "execution_date": execution_date,
        }

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return {"success": False, "error": str(e), "execution_date": execution_date}


def load_training_data(input_path: str) -> tuple[pl.DataFrame, pl.DataFrame, List[str]]:
    """
    Load training data from parquet files.

    Args:
        input_path: Path to the directory containing training data

    Returns:
        Tuple of (train_data, val_data, feature_columns)

    Raises:
        FileNotFoundError: If training data files are not found
    """
    # Check if files exist
    train_path = os.path.join(input_path, "train_data.parquet")
    val_path = os.path.join(input_path, "val_data.parquet")
    feature_path = os.path.join(input_path, "feature_columns.json")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data file not found: {val_path}")

    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature columns file not found: {feature_path}")

    # Load feature columns
    with open(feature_path, "r") as f:
        feature_columns = json.load(f)

    # Load training and validation data
    train_data = pl.read_parquet(train_path)
    val_data = pl.read_parquet(val_path)

    return train_data, val_data, feature_columns


def create_model(
    model_type: str, input_features: List[str], hyperparameters: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create a model of the specified type.

    Args:
        model_type: Type of model to create
        input_features: List of input feature names
        hyperparameters: Model-specific hyperparameters

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model type is not supported
    """
    if hyperparameters is None:
        hyperparameters = get_default_hyperparameters(model_type)

    if model_type == "gradient_boosting":
        from src.models.game_prediction.basic_model import BasicGamePredictionModel

        return BasicGamePredictionModel(input_features=input_features, **hyperparameters)
    elif model_type == "neural_network":
        from src.models.game_prediction.neural_model import NeuralGamePredictionModel

        return NeuralGamePredictionModel(input_features=input_features, **hyperparameters)
    elif model_type == "logistic_regression":
        from src.models.game_prediction.basic_model import LogisticRegressionModel

        return LogisticRegressionModel(input_features=input_features, **hyperparameters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_default_hyperparameters(model_type: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for the specified model type.

    Args:
        model_type: Type of model

    Returns:
        Dictionary of default hyperparameters
    """
    if model_type == "gradient_boosting":
        return {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 100, "subsample": 0.8}
    elif model_type == "neural_network":
        return {
            "learning_rate": 0.001,
            "hidden_layers": [64, 32],
            "dropout": 0.2,
            "batch_size": 32,
            "num_epochs": 50,
        }
    elif model_type == "logistic_regression":
        return {"learning_rate": 0.01, "regularization": 0.01, "max_iter": 1000}
    else:
        return {}
