---
title: NCAA Basketball Analytics - MLflow Usage Guide
description: Guide for using MLflow for experiment tracking and model registry in NCAA basketball analytics
---

# MLflow Usage Guide

This guide covers how to use MLflow for experiment tracking, model versioning, and model registry in the NCAA Basketball Analytics project.

## Overview

The project leverages MLflow for:

1. **Experiment Tracking**: Recording model parameters, metrics, and artifacts
2. **Model Registry**: Versioning and managing models
3. **Model Serving**: Deploying models for inference
4. **Reproducibility**: Capturing experiment context

## Setup and Configuration

### MLflow Server Setup

The project can use either a local MLflow server or a remote server. Configuration is managed through the `config/mlflow.yaml` file.

#### Local Server Configuration

For local development, you can run a local MLflow server:

```bash
# Start a local MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 127.0.0.1 \
  --port 5000
```

Update the configuration in `config/mlflow.yaml`:

```yaml
mlflow:
  tracking_uri: http://127.0.0.1:5000
  registry_uri: sqlite:///mlflow.db
  artifact_location: ./mlflow-artifacts
  experiment_name: ncaa_basketball_models
```

#### Remote Server Configuration

For team collaboration, configure a remote MLflow server:

```yaml
mlflow:
  tracking_uri: https://your-mlflow-server-url
  registry_uri: postgresql://user:password@host:port/database
  artifact_location: s3://your-bucket/mlflow-artifacts
  experiment_name: ncaa_basketball_models
```

### Initializing MLflow in Code

The project provides utilities for initializing MLflow:

```python
from src.models.mlflow.tracking import setup_mlflow_tracking, get_mlflow_client

# Set up MLflow tracking
setup_mlflow_tracking()

# Get MLflow client
client = get_mlflow_client()

# Access experiment
experiment = client.get_experiment_by_name("ncaa_basketball_models")
print(f"Experiment ID: {experiment.experiment_id}")
```

## Experiment Tracking

### Tracking Model Training

The framework provides the `MLflowTracker` class for tracking experiments:

```python
from src.models.mlflow.tracking import MLflowTracker
from src.models.training.trainer import ModelTrainer
from src.models.game_prediction.basic_model import BasicGamePredictionModel

# Create model
model = BasicGamePredictionModel(config=model_config)

# Create trainer
trainer = ModelTrainer(
    model=model,
    learning_rate=0.001,
    optimizer="adam"
)

# Initialize MLflow tracker
mlflow_tracker = MLflowTracker(
    experiment_name="game_prediction_models",
    run_name="basic_model_v1"
)

# Training with MLflow tracking
with mlflow_tracker:
    # Log hyperparameters
    mlflow_tracker.log_params(model.get_hyperparameters())
    mlflow_tracker.log_param("learning_rate", 0.001)

    # Train model and log metrics
    metrics_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100
    )

    # Log training metrics
    for epoch, metrics in enumerate(metrics_history):
        mlflow_tracker.log_metrics(metrics, step=epoch)

    # Log the final model
    mlflow_tracker.log_model(model, "game_prediction_model")

    # Log artifacts
    mlflow_tracker.log_artifact("learning_curves.png")
```

### Manual Tracking

For more fine-grained control, you can use MLflow's API directly:

```python
import mlflow
from src.models.mlflow.tracking import setup_mlflow_tracking

# Setup MLflow
setup_mlflow_tracking()

# Start a run
mlflow.start_run(run_name="custom_training_run")

# Log parameters
mlflow.log_param("model_type", "ensemble")
mlflow.log_param("num_models", 3)
mlflow.log_params({
    "hidden_size": 64,
    "dropout": 0.3,
    "activation": "relu"
})

# Log metrics
mlflow.log_metric("train_loss", 0.123)
mlflow.log_metric("val_loss", 0.145)
mlflow.log_metrics({
    "accuracy": 0.89,
    "precision": 0.92,
    "recall": 0.87
})

# Log artifacts
mlflow.log_artifact("feature_importance.png")

# End the run
mlflow.end_run()
```

### Logging Model Artifacts

Log models for later retrieval:

```python
import mlflow.pytorch
from src.models.mlflow.tracking import setup_mlflow_tracking

# Setup MLflow
setup_mlflow_tracking()

# Start a run
with mlflow.start_run(run_name="model_logging"):
    # Train your model
    # ...

    # Log the model
    mlflow.pytorch.log_model(
        model,
        artifact_path="models/game_prediction",
        registered_model_name="GamePredictionModel"
    )

    # Log additional artifacts
    mlflow.log_artifact("model_diagram.png", "diagrams")
```

## Model Registry

### Registering Models

Register models to track versions and stages:

```python
from src.models.mlflow.registry import register_model_from_run

# Register a model from a run
model_version = register_model_from_run(
    run_id="run_id_from_mlflow",
    model_name="GamePredictionModel",
    model_path="models/game_prediction"
)

print(f"Registered model version: {model_version.version}")
```

### Managing Model Versions

Get information about registered models:

```python
from src.models.mlflow.registry import (
    get_latest_model_version,
    list_model_versions,
    list_model_versions_with_stages
)

# Get the latest version of a model
latest_version = get_latest_model_version("GamePredictionModel")
print(f"Latest version: {latest_version.version}")

# List all versions of a model
versions = list_model_versions("GamePredictionModel")
for version in versions:
    print(f"Version {version.version}, stage: {version.current_stage}")

# List versions by stage
stage_versions = list_model_versions_with_stages(
    "GamePredictionModel",
    stages=["Production", "Staging"]
)
```

### Transitioning Model Stages

Move models through development stages:

```python
from src.models.mlflow.registry import transition_model_stage

# Transition a model to staging
transition_model_stage(
    model_name="GamePredictionModel",
    version=1,
    stage="Staging"
)

# Promote a model to production
transition_model_stage(
    model_name="GamePredictionModel",
    version=1,
    stage="Production"
)
```

### Loading Models from Registry

Load models for inference:

```python
from src.models.mlflow.registry import load_model_from_registry

# Load the latest production model
model = load_model_from_registry(
    model_name="GamePredictionModel",
    stage="Production"
)

# Load a specific version
model = load_model_from_registry(
    model_name="GamePredictionModel",
    version=2
)
```

## MLflow UI

The MLflow UI provides a visual interface for exploring experiments and models.

### Accessing the UI

Open the MLflow UI in your browser:

```bash
# If running locally
open http://127.0.0.1:5000
```

### Key UI Features

1. **Experiments Tab**: View all experiments and runs
2. **Run Comparison**: Compare multiple runs side-by-side
3. **Metric Visualization**: Plot metrics over time
4. **Artifact Viewer**: Explore model artifacts and plots
5. **Model Registry**: Manage model versions and stages

## Best Practices

### Experiment Organization

1. **Use meaningful experiment names** that reflect the purpose
2. **Group related runs** under the same experiment
3. **Use descriptive run names** with version information

Example naming scheme:
- Experiment: `game_prediction_models`
- Runs: `basic_v1`, `ensemble_v1`, `hyperparameter_tuning_v1`

### Parameter Tracking

1. **Log all hyperparameters** for reproducibility
2. **Include preprocessing parameters** that affect the model
3. **Track random seeds** for reproducible results
4. **Log environment information** like framework versions

### Metric Logging

1. **Log metrics at each epoch** during training
2. **Include both training and validation metrics**
3. **Log final evaluation metrics** on test data
4. **Track custom metrics** specific to basketball predictions

### Artifact Management

1. **Log model checkpoints** at regular intervals
2. **Save visualization plots** as artifacts
3. **Include feature importance diagrams**
4. **Store evaluation reports**

### Model Registry Workflow

Establish a clear workflow for model progression:

1. **None**: Initial model registration
2. **Staging**: Models under evaluation
3. **Production**: Models approved for use
4. **Archived**: Deprecated models

## Using MLflow in Scripts

### Training Script Integration

Example of MLflow integration in a training script:

```python
import argparse
import mlflow
from src.models.mlflow.tracking import setup_mlflow_tracking
from src.models.training import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="basic")
    parser.add_argument("--experiment-name", default="game_prediction")
    args = parser.parse_args()

    # Setup MLflow
    setup_mlflow_tracking()
    mlflow.set_experiment(args.experiment_name)

    # Start run with auto-logging
    mlflow.pytorch.autolog()

    with mlflow.start_run(run_name=f"{args.model_type}_model"):
        # Train model
        model, metrics = train_model(model_type=args.model_type)

        # Log additional metrics not captured by autolog
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "calibration_error": metrics["calibration_error"]
        })

        # Register model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=f"{args.model_type.capitalize()}GamePredictionModel"
        )

        print(f"Model trained and registered. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
```

### Evaluation Script Integration

Example of MLflow in an evaluation script:

```python
import argparse
import mlflow
from src.models.mlflow.registry import load_model_from_registry
from src.models.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", type=int)
    parser.add_argument("--stage", default="Production")
    args = parser.parse_args()

    # Load model from registry
    if args.model_version:
        model = load_model_from_registry(args.model_name, version=args.model_version)
    else:
        model = load_model_from_registry(args.model_name, stage=args.stage)

    # Evaluate model
    evaluation_results = evaluate_model(model)

    # Log evaluation results to a new run
    mlflow.set_experiment("model_evaluation")
    with mlflow.start_run(run_name=f"evaluation_{args.model_name}"):
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("model_version", args.model_version or "latest")
        mlflow.log_param("stage", args.stage)

        # Log metrics
        mlflow.log_metrics(evaluation_results["metrics"])

        # Log artifacts
        for plot_path in evaluation_results["plots"]:
            mlflow.log_artifact(plot_path)

        # Log evaluation report
        mlflow.log_artifact(evaluation_results["report_path"])

    print(f"Evaluation complete. Results logged to MLflow.")

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Connection errors**: Check that the MLflow server is running and tracking URI is correct
2. **Missing artifacts**: Verify artifact storage is properly configured
3. **Permission issues**: Ensure proper access to the storage location
4. **Database locks**: When using SQLite, ensure only one process accesses the database

### Debug Logging

Enable MLflow debug logging for troubleshooting:

```python
import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)
```

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch with MLflow Guide](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
