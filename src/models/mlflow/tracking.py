import mlflow
import os
import json
from typing import Dict, Optional, Union, Any
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from ..base import BaseModel, ModelConfig


def setup_mlflow_tracking(tracking_uri: Optional[str] = None) -> None:
    """
    Set up MLflow tracking URI.

    Args:
        tracking_uri: URI for MLflow tracking server
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def get_mlflow_client() -> Any:
    """
    Get an MLflow tracking client.

    Returns:
        MLflow tracking client
    """
    return mlflow.tracking.MlflowClient()


def setup_mlflow(
    tracking_uri: Optional[str] = None, experiment_name: str = "basketball_predictions"
) -> str:
    """
    Set up MLflow for experiment tracking.

    Args:
        tracking_uri: URI for MLflow tracking server
        experiment_name: Name of the MLflow experiment

    Returns:
        ID of the experiment
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Set the active experiment
    mlflow.set_experiment(experiment_name)

    return experiment_id


def log_model_params(params: Dict[str, Any]) -> None:
    """
    Log model parameters to MLflow.

    Args:
        params: Dictionary of parameter names and values
    """
    for name, value in params.items():
        # Handle various parameter types
        if isinstance(value, (list, tuple, np.ndarray)):
            # Convert lists to strings
            mlflow.log_param(name, str(value))
        elif isinstance(value, dict):
            # Convert dictionaries to JSON strings
            mlflow.log_param(name, json.dumps(value))
        else:
            # Log other types directly
            mlflow.log_param(name, value)


def log_model_metrics(metrics: Dict[str, Union[float, int]]) -> None:
    """
    Log model metrics to MLflow.

    Args:
        metrics: Dictionary of metric names and values
    """
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            mlflow.log_metric(name, value)


class MLflowTracker:
    """
    Class to manage MLflow tracking for model training and evaluation.

    This provides a unified interface for logging parameters, metrics,
    artifacts, and models to MLflow.
    """

    def __init__(
        self,
        experiment_name: str = "basketball_predictions",
        model_config: Optional[ModelConfig] = None,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            model_config: Model configuration for logging hyperparameters
            tracking_uri: URI for MLflow tracking server
            run_name: Name for the MLflow run
            tags: Dictionary of tags to attach to the run
        """
        self.experiment_name = experiment_name
        self.model_config = model_config
        self.tracking_uri = tracking_uri
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tags = tags or {}

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.active_run = None

    def start_run(self) -> None:
        """Start an MLflow run and log model configuration."""
        # Set the experiment
        mlflow.set_experiment(self.experiment_name)

        # Start the run
        self.active_run = mlflow.start_run(run_name=self.run_name)

        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

        # Log model configuration if provided
        if self.model_config:
            # Log model type as a tag
            mlflow.set_tag("model_type", self.model_config.model_type)

            # Log hyperparameters
            for key, value in self.model_config.hyperparameters.items():
                mlflow.log_param(key, value)

            # Log features as a parameter
            mlflow.log_param("features", str(self.model_config.features))

            # Log training parameters
            for key, value in self.model_config.training_params.items():
                mlflow.log_param(f"training_{key}", value)

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameter names and values
        """
        log_model_params(params)

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Step value for the metrics
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(name, value, step=step)

    def log_model(
        self, model: BaseModel, model_path: str, registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log a PyTorch model to MLflow.

        Args:
            model: Model to log
            model_path: Path to the saved model file
            registered_model_name: Name to register the model under (if any)
        """
        # Log the model file as an artifact
        mlflow.log_artifact(model_path)

        # Log the model version if available
        if hasattr(model, "get_version"):
            version = model.get_version()
            if version and hasattr(version, "version_id"):
                mlflow.set_tag("model_version", version.version_id)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact to MLflow.

        Args:
            local_path: Path to the file to log
            artifact_path: Path within the artifact directory
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_figure(
        self,
        figure: plt.Figure,
        artifact_file: str = "plot.png",
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure to log
            artifact_file: Filename for the artifact
            artifact_path: Path within the artifact directory
        """
        # Create a temporary directory if it doesn't exist
        tmp_dir = "/tmp/mlflow_figures"
        os.makedirs(tmp_dir, exist_ok=True)

        # Save the figure
        tmp_path = os.path.join(tmp_dir, artifact_file)
        figure.savefig(tmp_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        # Log the artifact
        self.log_artifact(tmp_path, artifact_path)

    def __enter__(self):
        """Start a new MLflow run when entering a context."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the MLflow run when exiting a context."""
        self.end_run()
