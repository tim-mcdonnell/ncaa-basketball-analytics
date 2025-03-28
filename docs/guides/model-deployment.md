---
title: NCAA Basketball Analytics - Model Deployment Guide
description: Guide for deploying and serving predictive models for NCAA basketball analytics
---

# Model Deployment Guide

This guide covers how to deploy trained models for inference in different environments, from development to production.

## Overview

The NCAA Basketball Analytics framework provides several deployment options:

1. **Local Inference**: For development and testing
2. **Batch Prediction**: For generating predictions in bulk
3. **API Deployment**: For real-time predictions
4. **Dashboard Integration**: For visualization in the analytics dashboard

## Prerequisites

Before deploying a model, ensure you have:

1. A trained and evaluated model
2. The model registered in the MLflow registry
3. Access to the deployment environment
4. Required environment variables and configurations

## Local Inference

### Loading Models

Load a model for local inference:

```python
from src.models.inference.model_loader import load_model

# Load a model from a local file
model = load_model("models/game_prediction_model.pt")

# Load a model from MLflow
model = load_model("mlflow://GamePredictionModel/Production")
```

### Making Predictions

Use the `GamePredictor` class for making predictions:

```python
from src.models.inference.predictor import GamePredictor

# Create predictor with loaded model
predictor = GamePredictor(model=model)

# Predict a single game
prediction = predictor.predict_game(
    home_team_id="MICH",
    away_team_id="OSU",
    game_date="2023-02-15"
)

print(f"Predicted spread: {prediction['spread']:.2f}")
print(f"Win probability: {prediction['win_probability']:.2f}")
```

### Batch Prediction

Process multiple games in batch:

```python
import polars as pl
from src.models.inference.predictor import GamePredictor

# Load game data
games_df = pl.read_csv("data/upcoming_games.csv")

# Create predictor
predictor = GamePredictor(model=model)

# Make batch predictions
predictions = predictor.predict_games(games_df)

# Save predictions
predictions.write_csv("data/predicted_games.csv")
```

## Batch Processing Pipeline

### Creating a Prediction DAG

Create an Airflow DAG for batch predictions:

```python
# airflow/dags/game_predictions_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.models.inference.batch import run_batch_predictions

default_args = {
    'owner': 'analytics',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'game_predictions',
    default_args=default_args,
    description='Generate game predictions',
    schedule_interval='0 2 * * *',  # Daily at 2:00 AM
)

generate_predictions = PythonOperator(
    task_id='generate_predictions',
    python_callable=run_batch_predictions,
    op_kwargs={
        'model_uri': 'mlflow://GamePredictionModel/Production',
        'output_path': 'data/predictions/{{ ds }}/predictions.csv',
    },
    dag=dag,
)
```

### Batch Prediction Script

Create a standalone script for batch predictions:

```python
# src/models/scripts/batch_predict.py
import argparse
import polars as pl
from src.models.inference.predictor import GamePredictor
from src.models.inference.model_loader import load_model
from src.data.game import get_upcoming_games

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True, help="Model URI (file path or MLflow URI)")
    parser.add_argument("--output-path", required=True, help="Path to save predictions")
    parser.add_argument("--days", type=int, default=7, help="Number of days to predict")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_uri)

    # Get upcoming games
    games_df = get_upcoming_games(days=args.days)

    # Create predictor
    predictor = GamePredictor(model=model)

    # Make predictions
    predictions = predictor.predict_games(games_df)

    # Save predictions
    predictions.write_csv(args.output_path)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()
```

## API Deployment

### Flask API

Create a simple Flask API for model serving:

```python
# src/api/prediction_api.py
from flask import Flask, request, jsonify
from src.models.inference.predictor import GamePredictor
from src.models.inference.model_loader import load_model

app = Flask(__name__)

# Load model at startup
model = load_model("mlflow://GamePredictionModel/Production")
predictor = GamePredictor(model=model)

@app.route("/predict/game", methods=["POST"])
def predict_game():
    data = request.json

    # Validate input
    required_fields = ["home_team_id", "away_team_id", "game_date"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # Make prediction
    try:
        prediction = predictor.predict_game(
            home_team_id=data["home_team_id"],
            away_team_id=data["away_team_id"],
            game_date=data["game_date"]
        )

        return jsonify({
            "home_team_id": data["home_team_id"],
            "away_team_id": data["away_team_id"],
            "game_date": data["game_date"],
            "spread": float(prediction["spread"]),
            "win_probability": float(prediction["win_probability"]),
            "confidence": float(prediction["confidence"])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### FastAPI Implementation

For a more modern API with automatic documentation:

```python
# src/api/fastapi_app.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
from typing import Optional

from src.models.inference.predictor import GamePredictor
from src.models.inference.model_loader import load_model

app = FastAPI(title="NCAA Basketball Prediction API")

# Load model at startup
model = load_model("mlflow://GamePredictionModel/Production")
predictor = GamePredictor(model=model)

class GameInput(BaseModel):
    home_team_id: str
    away_team_id: str
    game_date: date
    neutral_site: Optional[bool] = False

class PredictionOutput(BaseModel):
    home_team_id: str
    away_team_id: str
    game_date: date
    spread: float
    win_probability: float
    confidence: float

@app.post("/predict/game", response_model=PredictionOutput)
async def predict_game(game: GameInput):
    try:
        prediction = predictor.predict_game(
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
            game_date=str(game.game_date),
            neutral_site=game.neutral_site
        )

        return PredictionOutput(
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
            game_date=game.game_date,
            spread=float(prediction["spread"]),
            win_probability=float(prediction["win_probability"]),
            confidence=float(prediction["confidence"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=5000, reload=True)
```

## Docker Deployment

### Containerizing the API

Create a Dockerfile for the API:

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY config/ /app/config/

# Set environment variables
ENV MODEL_URI=mlflow://GamePredictionModel/Production
ENV MLFLOW_TRACKING_URI=https://your-mlflow-server
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Run the API
CMD ["python", "src/api/fastapi_app.py"]
```

### Building and Running the Container

```bash
# Build the image
docker build -t ncaa-prediction-api .

# Run the container
docker run -p 5000:5000 \
  -e MODEL_URI=mlflow://GamePredictionModel/Production \
  -e MLFLOW_TRACKING_URI=https://your-mlflow-server \
  -e MLFLOW_TRACKING_USERNAME=username \
  -e MLFLOW_TRACKING_PASSWORD=password \
  ncaa-prediction-api
```

## Dashboard Integration

### Prediction Component

Integrate predictions into the dashboard:

```python
# src/dashboard/components/prediction_component.py
import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import polars as pl

from src.models.inference.predictor import GamePredictor
from src.models.inference.model_loader import load_model

# Load model
model = load_model("mlflow://GamePredictionModel/Production")
predictor = GamePredictor(model=model)

# Create component
def create_prediction_component():
    return html.Div([
        html.H3("Game Prediction"),
        dbc.Row([
            dbc.Col([
                html.Label("Home Team"),
                dcc.Dropdown(id="home-team-dropdown", options=[])
            ], width=6),
            dbc.Col([
                html.Label("Away Team"),
                dcc.Dropdown(id="away-team-dropdown", options=[])
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Game Date"),
                dcc.DatePickerSingle(id="game-date-picker", date=datetime.now().date())
            ], width=6),
            dbc.Col([
                html.Label("Neutral Site"),
                dcc.RadioItems(
                    id="neutral-site-radio",
                    options=[
                        {"label": "Yes", "value": True},
                        {"label": "No", "value": False}
                    ],
                    value=False
                )
            ], width=6),
        ]),
        dbc.Button("Predict", id="predict-button", color="primary", className="mt-3"),
        html.Div(id="prediction-output", className="mt-4")
    ])

# Callback for prediction
@callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    Input("home-team-dropdown", "value"),
    Input("away-team-dropdown", "value"),
    Input("game-date-picker", "date"),
    Input("neutral-site-radio", "value"),
    prevent_initial_call=True
)
def update_prediction(n_clicks, home_team_id, away_team_id, game_date, neutral_site):
    if not all([home_team_id, away_team_id, game_date]):
        return html.Div("Please select both teams and a game date.", className="text-danger")

    # Make prediction
    prediction = predictor.predict_game(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date,
        neutral_site=neutral_site
    )

    # Format prediction for display
    spread = prediction["spread"]
    win_prob = prediction["win_probability"]

    # Determine favorite team
    favorite = home_team_id if spread > 0 else away_team_id
    underdog = away_team_id if spread > 0 else home_team_id
    abs_spread = abs(spread)

    # Create output
    return html.Div([
        html.H4("Prediction Results"),
        html.P([
            f"Favorite: ", html.Strong(favorite),
            f" by {abs_spread:.1f} points"
        ]),
        html.P([
            f"Win probability: ",
            html.Strong(f"{win_prob:.1%}")
        ]),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=[favorite, underdog],
                        y=[win_prob, 1-win_prob],
                        marker_color=["#007bff", "#dc3545"]
                    )
                ],
                layout=go.Layout(
                    title="Win Probability",
                    yaxis=dict(title="Probability", tickformat=".0%"),
                    height=300
                )
            )
        )
    ])
```

## Production Considerations

### Model Monitoring

Set up monitoring for deployed models:

```python
# src/models/monitoring/model_monitor.py
import logging
import time
import polars as pl
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model predictions and performance."""

    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        self.prediction_log = []
        self.performance_metrics = {
            "prediction_count": 0,
            "avg_latency_ms": 0,
            "error_count": 0
        }

    def log_prediction(self, inputs: Dict[str, Any], prediction: Dict[str, Any], latency_ms: float):
        """Log a single prediction."""
        # Log the prediction
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "model_version": self.version,
            "latency_ms": latency_ms,
            "inputs": inputs,
            "prediction": prediction
        }
        self.prediction_log.append(log_entry)

        # Update metrics
        self.performance_metrics["prediction_count"] += 1
        self.performance_metrics["avg_latency_ms"] = (
            (self.performance_metrics["avg_latency_ms"] * (self.performance_metrics["prediction_count"] - 1) + latency_ms) /
            self.performance_metrics["prediction_count"]
        )

        # Log to file
        logger.info(f"Prediction: {log_entry}")

    def log_error(self, inputs: Dict[str, Any], error: str):
        """Log a prediction error."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "model_version": self.version,
            "inputs": inputs,
            "error": error
        }
        self.prediction_log.append(log_entry)
        self.performance_metrics["error_count"] += 1

        # Log to file
        logger.error(f"Prediction error: {log_entry}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics

    def export_logs(self, output_path: str):
        """Export prediction logs to file."""
        logs_df = pl.DataFrame(self.prediction_log)
        logs_df.write_csv(output_path)
        logger.info(f"Exported prediction logs to {output_path}")

    def reset(self):
        """Reset metrics and logs."""
        self.prediction_log = []
        self.performance_metrics = {
            "prediction_count": 0,
            "avg_latency_ms": 0,
            "error_count": 0
        }
```

### Model Versioning

Implement proper model versioning in the API:

```python
# src/api/model_manager.py
import threading
import time
from typing import Dict, Any
import logging

from src.models.inference.model_loader import load_model
from src.models.inference.predictor import GamePredictor
from src.models.mlflow.registry import get_latest_model_version

logger = logging.getLogger(__name__)

class ModelManager:
    """Manage model versions for the API."""

    def __init__(self, model_name: str, stage: str = "Production"):
        self.model_name = model_name
        self.stage = stage
        self.predictors: Dict[str, GamePredictor] = {}
        self.current_version = None
        self.reload_lock = threading.Lock()
        self.last_reload_time = 0
        self.reload_interval = 300  # 5 minutes

        # Initial load
        self._reload_model()

    def _reload_model(self):
        """Reload the model from the registry."""
        with self.reload_lock:
            try:
                # Check latest version in registry
                latest_version = get_latest_model_version(self.model_name, self.stage)
                version_id = str(latest_version.version)

                # If we don't have this version, load it
                if version_id not in self.predictors:
                    logger.info(f"Loading model {self.model_name} version {version_id}")
                    model_uri = f"mlflow://{self.model_name}/{self.stage}"
                    model = load_model(model_uri)
                    self.predictors[version_id] = GamePredictor(model=model)

                # Update current version
                self.current_version = version_id
                self.last_reload_time = time.time()

                # Clean up old versions
                for version in list(self.predictors.keys()):
                    if version != version_id:
                        logger.info(f"Removing old model version {version}")
                        del self.predictors[version]

                logger.info(f"Current model version: {self.current_version}")
                return self.current_version
            except Exception as e:
                logger.error(f"Error reloading model: {e}")
                if not self.current_version:
                    raise  # Re-raise if we don't have any model

    def get_predictor(self) -> GamePredictor:
        """Get the current predictor."""
        # Check if we need to reload
        current_time = time.time()
        if current_time - self.last_reload_time > self.reload_interval:
            self._reload_model()

        return self.predictors[self.current_version]

    def predict_game(self, **kwargs) -> Dict[str, Any]:
        """Predict a game using current version."""
        return self.get_predictor().predict_game(**kwargs)
```

### API Authentication

Add authentication to the API:

```python
# src/api/auth.py
from functools import wraps
from flask import request, jsonify
import os
import jwt
import datetime

# Secret key from environment variable
SECRET_KEY = os.environ.get("API_SECRET_KEY", "")

def generate_token(user_id: str) -> str:
    """Generate a JWT token."""
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def token_required(f):
    """Decorator to require valid token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Check for token in headers
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"error": "Token is missing"}), 401

        try:
            # Decode token
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = data["user_id"]
        except:
            return jsonify({"error": "Token is invalid"}), 401

        return f(current_user, *args, **kwargs)

    return decorated
```

## Deployment Checklist

Before deploying to production, verify:

1. ✅ **Model Evaluation**: Model has been thoroughly evaluated on test data
2. ✅ **Versioning**: Model is properly versioned in MLflow registry
3. ✅ **Documentation**: API endpoints and parameters are documented
4. ✅ **Error Handling**: All potential errors are handled gracefully
5. ✅ **Logging**: Comprehensive logging is implemented
6. ✅ **Authentication**: API endpoints are properly secured
7. ✅ **Monitoring**: System for monitoring model performance is in place
8. ✅ **Scalability**: Deployment can handle expected load
9. ✅ **Rollback Plan**: Process for rolling back to previous version exists
10. ✅ **Testing**: All API endpoints have been tested

## Conclusion

This guide covered the essentials of deploying models in the NCAA Basketball Analytics framework. By following these patterns, you can deploy models for various use cases, from batch predictions to real-time API serving.
