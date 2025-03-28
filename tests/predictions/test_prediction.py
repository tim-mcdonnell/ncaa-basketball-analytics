"""
Test module for game prediction functionality.
"""

import pytest
import os
import json
from unittest.mock import MagicMock, patch

import polars as pl
import torch

# Import the function to test (will be implemented)
from src.predictions.prediction import generate_predictions


@pytest.fixture
def sample_prediction_data(tmp_path):
    """Create sample prediction data for testing."""
    # Create input and output directories
    input_dir = tmp_path / "prediction_data"
    output_dir = tmp_path / "predictions"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create feature columns
    feature_columns = ["win_percentage_diff", "points_per_game_diff", "predicted_point_diff"]

    # Save feature columns
    with open(os.path.join(input_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)

    # Create prediction data
    prediction_data = pl.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "home_team": ["TEAM1", "TEAM3", "TEAM5"],
            "away_team": ["TEAM2", "TEAM4", "TEAM6"],
            "date": ["2023-03-01", "2023-03-02", "2023-03-02"],
            "win_percentage_diff": [0.15, 0.08, -0.12],
            "points_per_game_diff": [5.2, 2.1, -3.8],
            "predicted_point_diff": [3.5, 1.2, -4.5],
        }
    )

    # Save prediction data
    prediction_data.write_parquet(os.path.join(input_dir, "prediction_data.parquet"))

    return {
        "input_path": str(input_dir),
        "output_path": str(output_dir),
        "feature_columns": feature_columns,
        "prediction_data": prediction_data,
    }


@patch("src.predictions.prediction.load_model_from_registry")
def test_generate_predictions_with_registry_model(mock_load_model, sample_prediction_data):
    """Test generating predictions using a model from the registry."""
    # Arrange
    input_path = sample_prediction_data["input_path"]
    output_path = sample_prediction_data["output_path"]

    # Create mock model that returns predictable predictions
    mock_model = MagicMock()
    mock_model.predict.return_value = torch.tensor([[0.75], [0.60], [0.35]])
    mock_model.config = MagicMock()
    mock_model.config.features = sample_prediction_data["feature_columns"]

    mock_load_model.return_value = mock_model

    # Act
    result = generate_predictions(
        input_path=input_path,
        output_path=output_path,
        model_stage="production",
        tracking_uri="sqlite:///mlflow.db",
        execution_date="2023-03-01",
    )

    # Assert
    assert result is not None
    assert result["success"] is True
    assert result["games_predicted"] == 3

    # Verify model was loaded from registry correctly
    mock_load_model.assert_called_once_with(
        model_name="ncaa_basketball_prediction",  # Default model name
        stage="production",
        tracking_uri="sqlite:///mlflow.db",
    )

    # Verify output file was created
    assert os.path.exists(os.path.join(output_path, "predictions.parquet"))

    # Load and verify predictions
    predictions_df = pl.read_parquet(os.path.join(output_path, "predictions.parquet"))
    assert len(predictions_df) == 3
    assert "game_id" in predictions_df.columns
    assert "home_team" in predictions_df.columns
    assert "away_team" in predictions_df.columns
    assert "win_probability" in predictions_df.columns
    assert "predicted_winner" in predictions_df.columns
    assert "prediction_date" in predictions_df.columns


@patch("src.predictions.prediction.load_model")
def test_generate_predictions_with_direct_model_path(mock_load_model, sample_prediction_data):
    """Test generating predictions using a model loaded directly from a path."""
    # Arrange
    input_path = sample_prediction_data["input_path"]
    output_path = sample_prediction_data["output_path"]
    model_path = "/path/to/model"

    # Create mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = torch.tensor([[0.75], [0.60], [0.35]])
    mock_model.config = MagicMock()
    mock_model.config.features = sample_prediction_data["feature_columns"]

    mock_load_model.return_value = mock_model

    # Act
    result = generate_predictions(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        execution_date="2023-03-01",
    )

    # Assert
    assert result["success"] is True
    assert result["games_predicted"] == 3

    # Verify model was loaded from path correctly
    mock_load_model.assert_called_once_with(model_path)


def test_generate_predictions_requires_model_source(sample_prediction_data):
    """Test that generate_predictions requires either model_path or model_stage."""
    # Arrange
    input_path = sample_prediction_data["input_path"]
    output_path = sample_prediction_data["output_path"]

    # Act
    result = generate_predictions(
        input_path=input_path,
        output_path=output_path,
        execution_date="2023-03-01",
        # No model_path or model_stage provided
    )

    # Assert
    assert result["success"] is False
    assert "error" in result
    assert "either model_path or model_stage" in result["error"].lower()


@patch("src.predictions.prediction.load_model_from_registry")
def test_generate_predictions_handles_missing_data(mock_load_model, tmp_path):
    """Test that generate_predictions handles missing prediction data."""
    # Arrange
    input_path = str(tmp_path / "nonexistent")
    output_path = str(tmp_path / "output")

    # Act
    result = generate_predictions(
        input_path=input_path,
        output_path=output_path,
        model_stage="production",
        tracking_uri="sqlite:///mlflow.db",
        execution_date="2023-03-01",
    )

    # Assert
    assert result["success"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()

    # Verify model was not loaded
    mock_load_model.assert_not_called()


@patch("src.predictions.prediction.load_model_from_registry")
def test_generate_predictions_handles_model_errors(mock_load_model, sample_prediction_data):
    """Test that generate_predictions handles errors during model loading."""
    # Arrange
    input_path = sample_prediction_data["input_path"]
    output_path = sample_prediction_data["output_path"]

    # Mock model loading error
    mock_load_model.side_effect = Exception("Model loading error")

    # Act
    result = generate_predictions(
        input_path=input_path,
        output_path=output_path,
        model_stage="production",
        tracking_uri="sqlite:///mlflow.db",
        execution_date="2023-03-01",
    )

    # Assert
    assert result["success"] is False
    assert "error" in result
    assert "Model loading error" in result["error"]
