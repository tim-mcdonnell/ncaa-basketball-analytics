---
title: NCAA Basketball Analytics - Model Training Guide
description: Guide for training and optimizing predictive models for NCAA basketball analytics
---

# Model Training Guide

This guide provides instructions for training predictive models using the NCAA Basketball Analytics framework. It covers dataset preparation, model configuration, training processes, and best practices.

## Prerequisites

Before training models, ensure you have:

1. Set up the development environment
2. Generated and processed the necessary feature data
3. Configured MLflow for tracking experiments

## Quick Start

### Training a Basic Game Prediction Model

The simplest way to train a model is using the training script:

```bash
python -m src.models.scripts.train_model --model-type=basic --features=game_prediction
```

This command:
1. Loads the default game prediction feature set
2. Creates a basic game prediction model
3. Trains the model with default hyperparameters
4. Logs metrics and artifacts to MLflow
5. Saves the trained model

### Training with Custom Configuration

For more control, you can provide a configuration file:

```bash
python -m src.models.scripts.train_model --config=config/models/custom_model_config.yaml
```

Example configuration file (`custom_model_config.yaml`):

```yaml
model:
  name: CustomGamePredictionModel
  type: basic
  hyperparameters:
    hidden_size: 128
    dropout: 0.2
    activation: relu
    layers: 3

training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 200
  early_stopping_patience: 20
  validation_split: 0.2
  optimizer: adam

features:
  set: game_prediction
  additional_features:
    - opponent_offensive_efficiency
    - team_defensive_rebounds
```

## Dataset Preparation

### Creating Training Datasets

The framework provides functions for creating properly formatted datasets:

```python
from src.models.training.dataset import GameDataset, create_train_val_test_split

# Load feature data
feature_data = load_feature_data("game_prediction")

# Create train/validation/test splits
train_data, val_data, test_data = create_train_val_test_split(
    feature_data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    time_column="game_date"
)

# Create datasets
train_dataset = GameDataset(
    train_data,
    target_column="spread",
    feature_columns=feature_list
)

val_dataset = GameDataset(
    val_data,
    target_column="spread",
    feature_columns=feature_list
)
```

### Data Loaders

Create PyTorch DataLoaders for efficient batch processing:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
```

## Model Training

### Creating a Model

Create a model instance with appropriate configuration:

```python
from src.models.base import ModelConfig
from src.models.game_prediction.basic_model import BasicGamePredictionModel

# Define model configuration
config = ModelConfig(
    name="GamePredictionModel",
    model_type="basic",
    hyperparameters={
        "hidden_size": 64,
        "dropout": 0.3,
        "activation": "relu"
    },
    features=feature_list
)

# Create model
model = BasicGamePredictionModel(config=config)
```

### Training with ModelTrainer

The `ModelTrainer` class handles the training process:

```python
from src.models.training.trainer import ModelTrainer

# Create trainer
trainer = ModelTrainer(
    model=model,
    learning_rate=0.001,
    optimizer="adam",
    criterion="mse"
)

# Train model
training_metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    early_stopping_patience=10,
    checkpoint_dir="checkpoints"
)
```

### MLflow Integration

To track experiments with MLflow:

```python
from src.models.mlflow.tracking import MLflowTracker

# Initialize MLflow tracker
mlflow_tracker = MLflowTracker(
    experiment_name="game_prediction_models",
    run_name="basic_model_v1"
)

# Start tracking
with mlflow_tracker:
    # Log hyperparameters
    mlflow_tracker.log_params(model.get_hyperparameters())

    # Train model and log metrics
    for epoch, metrics in enumerate(training_metrics):
        mlflow_tracker.log_metrics(metrics, step=epoch)

    # Log the final model
    mlflow_tracker.log_model(model, "game_prediction_model")
```

## Hyperparameter Optimization

### Grid Search

Perform grid search to find optimal hyperparameters:

```python
from src.models.training.hyperparameter_tuning import grid_search

# Define parameter grid
param_grid = {
    "hidden_size": [32, 64, 128],
    "dropout": [0.2, 0.3, 0.4],
    "learning_rate": [0.01, 0.001, 0.0001]
}

# Perform grid search
best_params, results = grid_search(
    model_class=BasicGamePredictionModel,
    param_grid=param_grid,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    early_stopping_patience=5
)

print(f"Best parameters: {best_params}")
```

### Cross-Validation

Evaluate models with cross-validation:

```python
from src.models.evaluation.cross_validation import KFoldCrossValidator

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

# Print cross-validation results
print(cv_results.get_summary())
```

## Training Best Practices

### Early Stopping

Early stopping prevents overfitting by halting training when validation metrics stop improving:

```python
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,  # Maximum epochs
    early_stopping_patience=20,  # Stop if no improvement for 20 epochs
    early_stopping_metric="val_loss",  # Metric to monitor
    early_stopping_mode="min"  # Lower is better for loss
)
```

### Learning Rate Scheduling

Adjust learning rate during training:

```python
trainer = ModelTrainer(
    model=model,
    learning_rate=0.001,
    optimizer="adam",
    lr_scheduler="reduce_on_plateau",
    lr_scheduler_params={
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6
    }
)
```

### Regularization

Prevent overfitting with regularization techniques:

```python
config = ModelConfig(
    name="GamePredictionModel",
    model_type="basic",
    hyperparameters={
        "hidden_size": 64,
        "dropout": 0.3,  # Dropout for regularization
        "weight_decay": 1e-5  # L2 regularization
    },
    features=feature_list
)
```

## Model Evaluation During Training

The trainer monitors and reports various metrics during training:

- Loss metrics (MSE, RMSE, MAE)
- Accuracy metrics (prediction accuracy, point spread accuracy)
- Validation performance

Example output:

```
Epoch 10/100
Training loss: 2.456, Validation loss: 2.611
Training accuracy: 0.712, Validation accuracy: 0.695
Point spread accuracy: 0.731
```

## Troubleshooting

### Overfitting

If the model performs well on training data but poorly on validation data:

1. Increase dropout rate
2. Add weight decay (L2 regularization)
3. Reduce model complexity
4. Collect more training data
5. Implement data augmentation

### Underfitting

If the model performs poorly on both training and validation data:

1. Increase model complexity
2. Train for more epochs
3. Reduce regularization
4. Use more powerful model architecture
5. Improve feature engineering

### Unstable Training

If training is unstable with fluctuating metrics:

1. Reduce learning rate
2. Use gradient clipping
3. Try different optimizers
4. Normalize input features
5. Check for label noise or data errors

## Next Steps

After training a model:

1. Evaluate it thoroughly with the evaluation framework
2. Register it in the MLflow model registry
3. Use it for predictions with the inference pipeline
