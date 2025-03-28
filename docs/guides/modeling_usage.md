---
title: Predictive Modeling Usage Guide
description: Examples for using the predictive modeling framework.
---

# Predictive Modeling Usage Guide

This guide provides examples for common tasks using the predictive modeling framework components.

## Evaluating Model Predictions

After training a model and obtaining predictions and actual target values, you can use the `EvaluationMetrics` class to calculate various performance metrics and generate plots.

```python
import torch
from src.models.evaluation.metrics import EvaluationMetrics
from src.models.evaluation.visualization import save_evaluation_plots

# Assume you have:
# - predictions: torch.Tensor (model outputs, e.g., probabilities or point spreads)
# - actuals: torch.Tensor (ground truth values)
# - features: torch.Tensor (input features for importance calculation)
# - feature_names: List[str] (names of the input features)
# - model: Your trained model instance (needed for feature importance)

# Example Data (replace with your actual data)
predictions = torch.rand(100, 1) # Example probabilities
actuals = (predictions > 0.5).float() # Example binary targets
features = torch.randn(100, 5)
feature_names = [f'feature_{i}' for i in range(5)]
# model = YourLoadedModel() # Load or use your trained model here

# --- Calculate Metrics ---

# Initialize the evaluator
evaluator = EvaluationMetrics(
    predictions=predictions,
    actuals=actuals,
    features=features,      # Optional, for feature importance
    feature_names=feature_names, # Optional, for feature importance
    # model=model           # Optional, for feature importance
)

# Calculate all metrics
all_metrics = evaluator.calculate_all_metrics()

# Get a formatted report
report = evaluator.get_report()

print("--- Evaluation Report ---")
import json
print(json.dumps(report, indent=2))

# Access specific metrics
accuracy = report['metrics']['accuracy']
rmse = report['metrics']['rmse']
print(f"\nAccuracy: {accuracy:.4f}")
print(f"RMSE: {rmse:.4f}")

if 'calibration' in report['metrics']:
    brier_score = report['metrics']['calibration']['brier_score']
    print(f"Brier Score: {brier_score:.4f}")

if 'feature_importance' in report['metrics']:
    print("\nFeature Importance:")
    # Sort importance for display
    sorted_importance = dict(sorted(report['metrics']['feature_importance'].items(), key=lambda item: item[1], reverse=True))
    for name, score in sorted_importance.items():
        print(f"  {name}: {score:.4f}")

# --- Generate and Save Plots ---

output_directory = "./evaluation_plots"
model_name = "example_model"

# Prepare metrics data for saving plots
# Note: This structure matches what save_evaluation_plots expects
plot_data = {
    # 'learning_curve_data': {...}, # Add if you have epoch-based metrics
    'feature_names': feature_names if 'feature_importance' in report['metrics'] else [],
    'importance_scores': list(report['metrics']['feature_importance'].values()) if 'feature_importance' in report['metrics'] else [],
    'calibration_curve': report['metrics'].get('calibration', {}).get('calibration_curve', {}),
    'confusion_matrix': report['metrics'].get('confusion_matrix'),
}

# Filter out None values or empty dicts before saving
plot_data_filtered = {k: v for k, v in plot_data.items() if v}

if plot_data_filtered:
    try:
        saved_plot_paths = save_evaluation_plots(
            metrics=plot_data_filtered,
            output_dir=output_directory,
            model_name=model_name
        )
        print(f"\nSaved evaluation plots to: {output_directory}")
        for plot_type, path in saved_plot_paths.items():
            print(f"  - {plot_type}: {path}")
    except Exception as e:
        print(f"\nError saving plots: {e}")
else:
    print("\nNo data available to generate plots.")

```

*Note: The `save_evaluation_plots` function requires specific keys (`learning_curve_data`, `feature_names`, `importance_scores`, `calibration_curve`, `confusion_matrix`) within the `metrics` dictionary passed to it.* The example above constructs this dictionary from the `EvaluationMetrics` report.

## Interacting with MLflow Model Registry

You can use the functions in `src.models.mlflow.registry` to manage models.

```python
from src.models.mlflow.registry import (
    register_model,
    get_latest_model_version,
    list_model_versions,
    load_registered_model
)

# --- Registering a Model ---

# Assume you have saved a model artifact during an MLflow run
run_id = "your_mlflow_run_id" # Replace with actual run ID
model_artifact_path = "model" # Path to artifact within the run
model_name = "NCAA_Game_Predictor"

model_uri_in_run = f"runs:/{run_id}/{model_artifact_path}"

try:
    registered_uri = register_model(
        model_path=model_uri_in_run,
        name=model_name,
        description="Initial version of the game predictor model.",
        tags={"framework": "pytorch", "type": "classification"}
    )
    print(f"Model registered successfully: {registered_uri}")

    # Example: Registering and transitioning to Staging
    # registered_uri_staging = register_model(
    #     model_path=model_uri_in_run,
    #     name=model_name,
    #     stage="Staging",
    #     description="Model candidate for staging."
    # )
    # print(f"Model registered to Staging: {registered_uri_staging}")

except Exception as e:
    print(f"Error registering model: {e}")

# --- Listing Model Versions ---

try:
    print(f"\nVersions for model '{model_name}':")
    versions = list_model_versions(name=model_name)
    for v in versions:
        print(f"  - Version: {v['version']}, Stage: {v['stage']}, Run ID: {v['run_id']}")

    print(f"\nProduction versions for model '{model_name}':")
    prod_versions = list_model_versions(name=model_name, stages=["Production"])
    for v in prod_versions:
        print(f"  - Version: {v['version']}, Stage: {v['stage']}")

except Exception as e:
    print(f"Error listing model versions: {e}")

# --- Getting Latest Version Details ---

try:
    latest_prod = get_latest_model_version(name=model_name, stage="Production")
    if latest_prod:
        print(f"\nLatest Production Version Details:")
        print(f"  Version: {latest_prod['version']}")
        print(f"  Run ID: {latest_prod['run_id']}")
        print(f"  Timestamp: {latest_prod['creation_timestamp']}")
        print(f"  Description: {latest_prod['description']}")
    else:
        print(f"\nNo Production version found for model '{model_name}'.")

except Exception as e:
    print(f"Error getting latest model version: {e}")

# --- Loading a Registered Model ---

try:
    # Load the latest Production model
    print(f"\nLoading latest Production model: '{model_name}'")
    loaded_model_prod = load_registered_model(name=model_name, stage="Production")
    print(f"Loaded model type: {type(loaded_model_prod).__name__}")

    # Load a specific version
    # specific_version = "1" # Replace with an actual version number
    # print(f"\nLoading version '{specific_version}' of model: '{model_name}'")
    # loaded_model_version = load_registered_model(name=model_name, version=specific_version)
    # print(f"Loaded model type: {type(loaded_model_version).__name__}")

except Exception as e:
    print(f"Error loading registered model: {e}")

```

This guide covers the basic usage of the evaluation and MLflow registry components. As the training components are developed, examples for training models will be added here.
