from .base import BaseModel, ModelConfig, ModelVersion
from .game_prediction import BasicGamePredictionModel, ModelEnsemble
from .training import GameDataset, create_train_val_test_split, create_data_loaders, ModelTrainer
from .evaluation import (
    calculate_accuracy,
    calculate_point_spread_accuracy,
    calculate_calibration_metrics,
    calculate_feature_importance,
    perform_cross_validation,
)
from .mlflow import setup_mlflow, MLflowTracker, register_model, get_latest_model_version
from .inference import GamePredictor, create_feature_vector, batch_predict


# Model registry for easy access to model classes
_MODEL_REGISTRY = {
    "BasicGamePredictionModel": BasicGamePredictionModel,
    "ModelEnsemble": ModelEnsemble,
}


def create_model(model_type: str, model_params: dict) -> BaseModel:
    """
    Create a model instance from a model type and parameters.

    Args:
        model_type: Name of the model class to create
        model_params: Dictionary of parameters to pass to the model constructor

    Returns:
        Instance of the specified model class
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available models: {list(_MODEL_REGISTRY.keys())}"
        )

    model_class = _MODEL_REGISTRY[model_type]
    return model_class(**model_params)


def load_model(model_path: str) -> BaseModel:
    """
    Load a model from a file.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded model
    """
    import torch
    import os

    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Determine model type
    model_config = checkpoint.get("config", {})
    model_type = model_config.get("name", "BasicGamePredictionModel")

    # Create and load the model
    if model_type == "ModelEnsemble":
        # For ensembles, we need to load the individual models
        from .game_prediction.ensemble import ModelEnsemble

        return ModelEnsemble.load(model_path)
    else:
        # For basic models
        if model_type in _MODEL_REGISTRY:
            model_class = _MODEL_REGISTRY[model_type]
            return model_class.load(model_path)
        else:
            raise ValueError(f"Unknown model type in checkpoint: {model_type}")


__all__ = [
    # Base
    "BaseModel",
    "ModelConfig",
    "ModelVersion",
    # Game prediction models
    "BasicGamePredictionModel",
    "ModelEnsemble",
    # Training
    "GameDataset",
    "create_train_val_test_split",
    "create_data_loaders",
    "ModelTrainer",
    # Evaluation
    "calculate_accuracy",
    "calculate_point_spread_accuracy",
    "calculate_calibration_metrics",
    "calculate_feature_importance",
    "perform_cross_validation",
    # MLflow
    "setup_mlflow",
    "MLflowTracker",
    "register_model",
    "get_latest_model_version",
    # Inference
    "GamePredictor",
    "create_feature_vector",
    "batch_predict",
    # Model factories
    "create_model",
    "load_model",
]
