"""
Module for generating predictions for NCAA basketball games.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import polars as pl
import torch

from src.models.load import load_model, load_model_from_registry


def generate_predictions(
    input_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    model_stage: Optional[str] = None,
    model_name: str = "ncaa_basketball_prediction",
    tracking_uri: str = "sqlite:///mlflow.db",
    execution_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate predictions for upcoming games using a trained model.

    Args:
        input_path: Path to directory containing prediction data
        output_path: Path to directory to save predictions
        model_path: Path to saved model (alternative to model_stage)
        model_stage: Model stage in registry (alternative to model_path)
        model_name: Name of the model in registry
        tracking_uri: MLflow tracking URI
        execution_date: Execution date in YYYY-MM-DD format

    Returns:
        Dict with success status, number of games predicted, and any error information
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting prediction generation with input from {input_path}")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Set execution date if not provided
    if execution_date is None:
        execution_date = datetime.now().strftime("%Y-%m-%d")

    # Validate input arguments
    if model_path is None and model_stage is None:
        error_msg = "Either model_path or model_stage must be provided"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "games_predicted": 0}

    try:
        # Check if prediction data exists
        prediction_data_path = os.path.join(input_path, "prediction_data.parquet")
        feature_columns_path = os.path.join(input_path, "feature_columns.json")

        if not os.path.exists(prediction_data_path) or not os.path.exists(feature_columns_path):
            error_msg = f"Prediction data not found at {prediction_data_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "games_predicted": 0}

        # Load prediction data
        logger.info(f"Loading prediction data from {prediction_data_path}")
        prediction_data = pl.read_parquet(prediction_data_path)

        # If no games to predict, return early with empty predictions file
        if len(prediction_data) == 0:
            logger.info("No games to predict")
            empty_predictions = pl.DataFrame(
                {
                    "game_id": [],
                    "home_team": [],
                    "away_team": [],
                    "win_probability": [],
                    "predicted_winner": [],
                    "prediction_date": [],
                }
            )
            empty_predictions.write_parquet(os.path.join(output_path, "predictions.parquet"))
            return {"success": True, "games_predicted": 0}

        # Load model
        logger.info("Loading model")
        if model_path:
            model = load_model(model_path)
            logger.info(f"Model loaded from path: {model_path}")
        else:
            model = load_model_from_registry(
                model_name=model_name, stage=model_stage, tracking_uri=tracking_uri
            )
            logger.info(f"Model loaded from registry: {model_name} (stage: {model_stage})")

        # Extract feature columns from model if available
        try:
            with open(feature_columns_path, "r") as f:
                feature_columns = json.load(f)
                logger.info(f"Loaded {len(feature_columns)} feature columns")
        except Exception as e:
            error_msg = f"Failed to load feature columns: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "games_predicted": 0}

        # Prepare features for prediction
        features = prediction_data.select(feature_columns)

        # Convert to numpy for prediction
        features_np = features.to_numpy()

        # Generate predictions
        logger.info(f"Generating predictions for {len(prediction_data)} games")

        # Convert to tensor for model prediction
        features_tensor = torch.tensor(features_np, dtype=torch.float32)

        with torch.no_grad():
            predictions = model.predict(features_tensor)

        # Convert predictions to numpy
        win_probabilities = predictions.numpy().flatten()

        # Create predictions dataframe
        predictions_df = pl.DataFrame(
            {
                "game_id": prediction_data["game_id"],
                "home_team": prediction_data["home_team"],
                "away_team": prediction_data["away_team"],
                "win_probability": win_probabilities.tolist(),
                "predicted_winner": prediction_data["home_team"].zip_with(
                    prediction_data["away_team"],
                    win_probabilities,
                    lambda home, away, prob: home if prob >= 0.5 else away,
                ),
                "prediction_date": [execution_date] * len(prediction_data),
            }
        )

        # Save predictions
        output_file = os.path.join(output_path, "predictions.parquet")
        logger.info(f"Saving predictions to {output_file}")
        predictions_df.write_parquet(output_file)

        return {
            "success": True,
            "games_predicted": len(prediction_data),
            "output_file": output_file,
        }

    except Exception as e:
        error_msg = f"Error generating predictions: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "games_predicted": 0}
