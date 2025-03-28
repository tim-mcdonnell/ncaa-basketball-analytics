---
title: NCAA Basketball Analytics - Model Evaluation Guide
description: Guide for evaluating and analyzing predictive model performance in NCAA basketball analytics
---

# Model Evaluation Guide

This guide covers the tools and techniques available in the NCAA Basketball Analytics framework for evaluating model performance, analyzing prediction quality, and comparing different models.

## Overview

The model evaluation framework provides:

1. **Comprehensive metrics** for assessing prediction accuracy
2. **Cross-validation** for robust performance estimation
3. **Visualization tools** for performance analysis
4. **Feature importance** calculation
5. **Model comparison** utilities

## Quick Start

### Basic Model Evaluation

To quickly evaluate a trained model:

```bash
python -m src.models.scripts.evaluate_model --model-id=latest --dataset=test
```

This command:
1. Loads the latest registered model
2. Evaluates it on the test dataset
3. Calculates standard metrics
4. Generates visualizations
5. Outputs a performance report

### Comprehensive Evaluation

For more detailed evaluation:

```bash
python -m src.models.scripts.evaluate_model \
  --model-id=model_version_id \
  --dataset=test \
  --cross-validation=true \
  --n-folds=5 \
  --metrics=all \
  --visualize=true \
  --save-report=true
```

## Evaluation Metrics

### Core Metrics

The framework calculates several metrics to assess model performance:

#### Classification Metrics

For binary win/loss prediction:

```python
from src.models.evaluation.metrics import calculate_prediction_accuracy

# Get model predictions
predictions = model.predict(test_data)
actual_results = test_data["result"]

# Calculate accuracy
accuracy = calculate_prediction_accuracy(
    predictions,
    actual_results,
    threshold=0.5  # Threshold for binary classification
)

print(f"Prediction accuracy: {accuracy:.4f}")
```

#### Regression Metrics

For point spread prediction:

```python
from src.models.evaluation.metrics import calculate_point_spread_accuracy

# Calculate point spread accuracy
accuracy, details = calculate_point_spread_accuracy(
    y_true=actual_spreads,
    y_pred=predicted_spreads,
    return_details=True
)

print(f"Point spread accuracy: {accuracy:.4f}")
```

#### Calibration Metrics

For assessing probability calibration:

```python
from src.models.evaluation.metrics import calculate_calibration_metrics

# Calculate calibration metrics
calibration = calculate_calibration_metrics(
    y_true=actual_results,
    y_pred=win_probabilities,
    n_bins=10
)

print(f"Calibration error: {calibration['calibration_error']:.4f}")
print(f"Brier score: {calibration['brier_score']:.4f}")
```

### Comprehensive Metrics with EvaluationMetrics

The `EvaluationMetrics` class provides a comprehensive metrics calculation:

```python
from src.models.evaluation.metrics import EvaluationMetrics

# Create evaluation metrics
metrics = EvaluationMetrics(
    predictions=model_predictions,
    actuals=actual_values,
    features=feature_values,
    feature_names=feature_names,
    model=model
)

# Calculate all metrics
results = metrics.calculate_all_metrics()

# Get formatted report
report = metrics.get_report()
print(report)
```

## Cross-Validation

### Time Series Cross-Validation

For time-dependent data like basketball games:

```python
from src.models.evaluation.cross_validation import TimeSeriesSplit

# Create time series splitter
ts_split = TimeSeriesSplit(
    n_splits=5,
    test_size=0.2
)

# Generate splits
for train_idx, test_idx in ts_split.split(data, time_column="game_date"):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    # Train and evaluate model on this split
    # ...
```

### K-Fold Cross-Validation

For comprehensive evaluation:

```python
from src.models.evaluation.cross_validation import KFoldCrossValidator
from src.models.base import ModelConfig
from src.models.game_prediction.basic_model import BasicGamePredictionModel

# Define model configuration
config = ModelConfig(
    name="GamePredictionModel",
    model_type="basic",
    hyperparameters={
        "hidden_size": 64,
        "dropout": 0.3
    },
    features=feature_list
)

# Create cross-validator
cv = KFoldCrossValidator(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Run cross-validation
cv_results = cv.validate(
    model_class=BasicGamePredictionModel,
    model_config=config,
    feature_data=feature_data,
    target_column="spread",
    feature_columns=feature_list
)

# Print summary
print(cv_results.get_summary())
```

### Cross-Validation Results

The `CrossValidationResults` class provides aggregated metrics:

```python
# Get average metrics across folds
avg_metrics = cv_results.get_average_metrics()
print(f"Average accuracy: {avg_metrics['accuracy']:.4f}")
print(f"Average MSE: {avg_metrics['mse']:.4f}")

# Get metric distributions
metric_distributions = cv_results.get_metric_distributions()
print(f"Accuracy range: {metric_distributions['accuracy']['min']:.4f} - {metric_distributions['accuracy']['max']:.4f}")

# Print detailed summary
summary = cv_results.get_summary()
print(summary)
```

## Visualization

### Learning Curves

Plot training and validation metrics:

```python
from src.models.evaluation.visualization import plot_learning_curves

# Plot learning curves
fig = plot_learning_curves(
    training_metrics=training_history,
    metric="loss",
    title="Model Training Loss",
    figsize=(10, 6)
)

# Save the figure
fig.savefig("learning_curves.png")
```

### Feature Importance

Visualize feature importance:

```python
from src.models.evaluation.visualization import plot_feature_importance

# Calculate feature importance
importance = calculate_feature_importance(
    model=model,
    features=X_test,
    feature_names=feature_names
)

# Plot feature importance
fig = plot_feature_importance(
    importance=importance,
    top_n=10,
    title="Top 10 Important Features",
    figsize=(12, 8)
)

# Save the figure
fig.savefig("feature_importance.png")
```

### Calibration Curve

Assess probability calibration:

```python
from src.models.evaluation.visualization import plot_calibration_curve

# Plot calibration curve
fig = plot_calibration_curve(
    calibration_data=calibration,
    title="Win Probability Calibration",
    figsize=(10, 6)
)

# Save the figure
fig.savefig("calibration_curve.png")
```

### Confusion Matrix

Visualize classification performance:

```python
from src.models.evaluation.visualization import plot_confusion_matrix

# Plot confusion matrix
fig = plot_confusion_matrix(
    y_true=actual_results,
    y_pred=predicted_results,
    labels=["Loss", "Win"],
    title="Game Prediction Confusion Matrix",
    figsize=(8, 8)
)

# Save the figure
fig.savefig("confusion_matrix.png")
```

### Saving All Evaluation Plots

Save a complete set of evaluation plots:

```python
from src.models.evaluation.visualization import save_evaluation_plots

# Generate and save all evaluation plots
plots_dir = save_evaluation_plots(
    evaluation_metrics=metrics,
    training_history=training_metrics,
    output_dir="evaluation_plots",
    prefix="model_v1_"
)

print(f"Evaluation plots saved to: {plots_dir}")
```

## Model Comparison

### Comparing Multiple Models

To compare multiple models:

```python
from src.models.evaluation.comparison import compare_models

# Load models
model1 = load_model("model1_path")
model2 = load_model("model2_path")
model3 = load_model("model3_path")

# Define models to compare
models = {
    "basic": model1,
    "advanced": model2,
    "ensemble": model3
}

# Compare models
comparison = compare_models(
    models=models,
    test_data=test_data,
    target_column="spread",
    metrics=["accuracy", "mse", "point_spread_accuracy", "calibration_error"]
)

# Print comparison table
print(comparison.get_table())

# Get the best model by a specific metric
best_model_name = comparison.get_best_model(metric="accuracy")
print(f"Best model by accuracy: {best_model_name}")
```

### Visualizing Model Comparison

Create comparison charts:

```python
from src.models.evaluation.visualization import plot_model_comparison

# Plot model comparison
fig = plot_model_comparison(
    comparison_results=comparison,
    metrics=["accuracy", "mse", "calibration_error"],
    title="Model Performance Comparison",
    figsize=(12, 8)
)

# Save the figure
fig.savefig("model_comparison.png")
```

## Advanced Evaluation

### Bootstrapping Performance Metrics

For confidence intervals on metrics:

```python
from src.models.evaluation.bootstrap import bootstrap_metrics

# Calculate bootstrapped metrics
bootstrap_results = bootstrap_metrics(
    model=model,
    data=test_data,
    target_column="spread",
    feature_columns=feature_list,
    n_samples=1000,
    metrics=["accuracy", "mse"]
)

# Print confidence intervals
for metric, stats in bootstrap_results.items():
    print(f"{metric}: {stats['mean']:.4f} ({stats['ci_lower']:.4f} - {stats['ci_upper']:.4f})")
```

### Evaluating Different Game Subsets

Evaluate performance on different subsets:

```python
from src.models.evaluation.subset import evaluate_on_subsets

# Define subsets
subsets = {
    "home_games": test_data["is_home"] == True,
    "away_games": test_data["is_home"] == False,
    "conference_games": test_data["is_conference"] == True,
    "non_conference_games": test_data["is_conference"] == False
}

# Evaluate on subsets
subset_results = evaluate_on_subsets(
    model=model,
    data=test_data,
    subsets=subsets,
    target_column="spread",
    feature_columns=feature_list,
    metrics=["accuracy", "mse"]
)

# Print subset performance
for subset_name, metrics in subset_results.items():
    print(f"{subset_name}: Accuracy = {metrics['accuracy']:.4f}, MSE = {metrics['mse']:.4f}")
```

## Creating Evaluation Reports

Generate comprehensive HTML reports:

```python
from src.models.evaluation.report import create_evaluation_report

# Create evaluation report
report_path = create_evaluation_report(
    model=model,
    training_history=training_metrics,
    test_data=test_data,
    target_column="spread",
    feature_columns=feature_list,
    output_path="reports/model_evaluation.html",
    include_plots=True
)

print(f"Evaluation report saved to: {report_path}")
```

## Best Practices

1. **Always evaluate on held-out data** not used during training
2. **Use cross-validation** for more robust performance estimates
3. **Compare against baseline models** to assess improvement
4. **Visualize results** to better understand model behavior
5. **Analyze feature importance** to ensure model is using reasonable signals
6. **Evaluate on specific subsets** to identify weaknesses
7. **Calculate confidence intervals** for important metrics
8. **Document evaluation results** for future reference
