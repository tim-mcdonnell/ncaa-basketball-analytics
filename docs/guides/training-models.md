# Training Prediction Models

This guide explains how to train and evaluate prediction models in the NCAA Basketball Analytics system.

## Model Training Framework

The project uses a structured approach to model training, following the architecture defined in the [Model Training](../architecture/model-training.md) document.

## Model Types

The system supports several types of prediction models:

- **Game Outcome Models**: Predict win/loss or score margin
- **Player Performance Models**: Predict player statistics
- **Team Performance Models**: Predict team statistics

## Training a New Model

### 1. Define Model Requirements

First, clearly define:

- The prediction target
- Features required for prediction
- Evaluation metrics
- Use cases for the model

### 2. Prepare Training Data

Use the feature engineering framework to prepare your training data:

```python
from src.features.registry import get_feature_registry
from src.data.storage import get_db_connection
import polars as pl

# Get database connection
db_conn = get_db_connection()

# Query base data
games_query = """
SELECT g.game_id, g.season_id, g.game_date, g.home_team_id, g.away_team_id,
       g.home_score, g.away_score, 
       CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win
FROM fact_games g
WHERE g.status = 'completed'
ORDER BY g.game_date
"""
games_df = pl.from_arrow(db_conn.execute(games_query).arrow())

# Get feature registry
registry = get_feature_registry()

# Get features
features_to_use = [
    "team_scoring_efficiency",
    "team_defensive_efficiency",
    "team_pace",
    "team_shooting_pct_trend",
    # More features...
]

# Load features
feature_data = {}
for feature_name in features_to_use:
    feature = registry.get_feature(feature_name)
    feature_data[feature_name] = feature.load_latest(db_conn)

# Join features to create training data
# [Implementation details for your specific case]
```

### 3. Implement Model Training

Create a training script:

```python
# src/models/training.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import mlflow
import json

def train_game_outcome_model(
    features_df,
    config,
    model_name="game_outcome_prediction",
    model_version="1.0"
):
    """
    Train a game outcome prediction model
    
    Args:
        features_df: DataFrame with features and target
        config: Model configuration
        model_name: Name for the model
        model_version: Version string
        
    Returns:
        Trained model and evaluation metrics
    """
    # Split data
    X = features_df.drop(["game_id", "home_win"], axis=1)
    y = features_df["home_win"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.training.test_size,
        random_state=config.training.random_state
    )
    
    # Start MLflow run
    mlflow.start_run(run_name=f"{model_name}_v{model_version}")
    
    # Log parameters
    mlflow.log_params({
        "model_type": "lightgbm",
        "test_size": config.training.test_size,
        "random_state": config.training.random_state,
        "features": list(X.columns),
        **config.model_params.lightgbm.dict()
    })
    
    # Train model
    model = lgb.LGBMClassifier(
        **config.model_params.lightgbm.dict()
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_pred_proba),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.lightgbm.log_model(
        model, 
        artifact_path="model",
        registered_model_name=model_name
    )
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    mlflow.log_table(
        feature_importance.to_dict(orient="records"),
        artifact_file="feature_importance.json"
    )
    
    # End run
    mlflow.end_run()
    
    return model, metrics
```

### 4. Create an Airflow DAG for Training

Set up an Airflow DAG to schedule model training:

```python
# airflow/dags/model_training/train_game_outcome_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.models.training import train_game_outcome_model
from src.config.settings import load_config

config = load_config()

default_args = {
    'owner': 'ncaa_analytics',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'train_game_outcome_model',
    default_args=default_args,
    description='Train game outcome prediction model',
    schedule_interval='0 4 * * 0',  # Weekly on Sunday
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['model_training', 'game_prediction'],
) as dag:
    
    # Define training task
    train_model_task = PythonOperator(
        task_id='train_game_outcome_model',
        python_callable=train_game_outcome_model,
        op_kwargs={
            'config': config,
            'model_name': 'game_outcome_prediction',
            'model_version': datetime.now().strftime("%Y%m%d")
        }
    )
```

### 5. Evaluate and Monitor Models

Set up model evaluation processes:

```python
# src/models/evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import json

def evaluate_classification_model(
    model,
    X_test,
    y_test,
    threshold=0.5,
    model_name=None,
    run_id=None
):
    """
    Evaluate a binary classification model
    
    Args:
        model: Trained model with predict_proba method
        X_test: Test features
        y_test: True labels
        threshold: Classification threshold
        model_name: Name for mlflow logging
        run_id: MLflow run ID for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Generate predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "log_loss": log_loss(y_test, y_pred_proba),
        "threshold": threshold
    }
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # If mlflow run is provided, log results
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix as figure
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.close()
            
            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
    
    return metrics
```

## Model Deployment

Once a model is trained and evaluated, register it in MLflow and update the prediction pipeline:

```python
# src/models/prediction.py
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def load_latest_model(model_name: str):
    """
    Load the latest version of a registered model
    
    Args:
        model_name: Name of registered model
        
    Returns:
        Loaded model
    """
    latest_model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
    return latest_model

def predict_game_outcomes(
    upcoming_games: pd.DataFrame,
    features: Dict[str, pd.DataFrame],
    model_name: str = "game_outcome_prediction"
) -> pd.DataFrame:
    """
    Generate predictions for upcoming games
    
    Args:
        upcoming_games: DataFrame of upcoming games
        features: Dictionary of feature DataFrames
        model_name: Name of model to use
        
    Returns:
        DataFrame with game predictions
    """
    # Load model
    model = load_latest_model(model_name)
    
    # Prepare features for prediction
    # [Implementation details for your specific case]
    
    # Make predictions
    predictions = model.predict(X_pred)
    
    # Add predictions to games
    results = upcoming_games.copy()
    results["win_probability"] = predictions
    
    return results
```

## Best Practices

- Document the model methodology and features used
- Track all experiments using MLflow
- Regularly evaluate model performance
- Compare new models against existing models before deployment
- Test models thoroughly with different data splits
- Monitor model performance over time
