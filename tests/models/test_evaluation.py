"""
Test module for model evaluation functionality.
"""

import pytest
import os
import json
import polars as pl
import torch
from unittest.mock import MagicMock, patch

# Import the function to test (will be implemented)
from src.models.evaluation import evaluate_model


@pytest.fixture
def sample_model_and_data(tmp_path):
    """Create a sample model and test data for evaluation."""
    # Create directories
    model_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Create feature columns
    feature_columns = ["win_percentage_diff", "points_per_game_diff", "predicted_point_diff"]

    # Mock model config
    model_config = {
        "model_type": "gradient_boosting",
        "features": feature_columns,
        "hyperparameters": {"learning_rate": 0.05, "max_depth": 5},
        "training_params": {"training_date": "2023-01-01"},
        "version": {"major": 1, "minor": 0, "patch": 0},
    }

    # Save model config
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f)

    # Create test data
    test_data = pl.DataFrame(
        {
            "game_id": [f"G{i}" for i in range(1, 11)],
            "home_team": [f"TEAM{i}" for i in range(1, 11)],
            "away_team": [f"TEAM{i+10}" for i in range(1, 11)],
            "date": ["2023-01-03"] * 10,
            "home_win": [True, False, True, False, True, False, True, False, True, False],
            "point_diff": [5.0, -3.0, 4.0, -2.0, 6.0, -1.0, 3.0, -4.0, 2.0, -5.0],
            "win_percentage_diff": [0.2, -0.1, 0.15, -0.05, 0.25, -0.15, 0.1, -0.2, 0.05, -0.25],
            "points_per_game_diff": [3.5, -2.8, 2.5, -2.0, 4.0, -1.5, 3.0, -3.5, 1.5, -4.0],
            "predicted_point_diff": [4.2, -3.1, 3.5, -1.8, 4.5, -2.2, 2.8, -3.8, 1.2, -4.5],
        }
    )

    # Save test data
    test_data.write_parquet(os.path.join(data_dir, "test_data.parquet"))

    return {
        "model_path": str(model_dir),
        "test_data_path": os.path.join(data_dir, "test_data.parquet"),
        "feature_columns": feature_columns,
        "test_data": test_data,
    }


@patch("src.models.evaluation.load_model")
def test_evaluate_model_calculates_correct_metrics(mock_load_model, sample_model_and_data):
    """Test that evaluate_model calculates and returns the expected metrics."""
    # Arrange
    model_path = sample_model_and_data["model_path"]
    test_data_path = sample_model_and_data["test_data_path"]

    # Create mock model with predictable behavior
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda x: torch.tensor(
        [[0.8], [0.2], [0.75], [0.3], [0.9], [0.1], [0.7], [0.25], [0.6], [0.15]]
    )
    mock_model.config = MagicMock()
    mock_model.config.features = sample_model_and_data["feature_columns"]
    mock_load_model.return_value = mock_model

    # Act
    result = evaluate_model(
        model_path=model_path, test_data_path=test_data_path, execution_date="2023-01-03"
    )

    # Assert
    assert result is not None
    assert result["success"] is True

    # Verify metrics are calculated correctly
    assert "metrics" in result
    metrics = result["metrics"]

    # With our mock predictions and true values, we expect:
    # - 7/10 correct win predictions (70% accuracy)
    # - AUC should be high (close to 1.0)
    # - MSE should be relatively low
    assert "accuracy" in metrics
    assert (
        0.6 <= metrics["accuracy"] <= 0.8
    ), f"Expected accuracy around 0.7, got {metrics['accuracy']}"

    assert "auc" in metrics
    assert metrics["auc"] > 0.7, f"Expected AUC > 0.7, got {metrics['auc']}"

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics


@patch("src.models.evaluation.MLflowTracker")
@patch("src.models.evaluation.load_model")
def test_evaluate_model_logs_to_mlflow(
    mock_load_model, mock_mlflow_tracker_class, sample_model_and_data
):
    """Test that evaluate_model logs evaluation metrics to MLflow."""
    # Arrange
    model_path = sample_model_and_data["model_path"]
    test_data_path = sample_model_and_data["test_data_path"]
    tracking_uri = "sqlite:///mlflow.db"

    # Create mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = torch.rand(10, 1)
    mock_model.config = MagicMock()
    mock_model.config.features = sample_model_and_data["feature_columns"]
    mock_model.config.model_type = "gradient_boosting"
    mock_load_model.return_value = mock_model

    # Create mock MLflow tracker
    mock_tracker = MagicMock()
    mock_mlflow_tracker_class.return_value = mock_tracker

    # Act
    result = evaluate_model(
        model_path=model_path,
        test_data_path=test_data_path,
        tracking_uri=tracking_uri,
        execution_date="2023-01-03",
    )

    # Assert
    assert result["success"] is True

    # Verify MLflow was used correctly
    mock_mlflow_tracker_class.assert_called_once_with(tracking_uri=tracking_uri)
    mock_tracker.start_run.assert_called_once()
    mock_tracker.log_params.assert_called_once()
    mock_tracker.log_metrics.assert_called_once()

    # Verify metrics were logged
    metrics_arg = mock_tracker.log_metrics.call_args[0][0]
    assert "accuracy" in metrics_arg
    assert "auc" in metrics_arg
    assert "mse" in metrics_arg
    assert "rmse" in metrics_arg
    assert "mae" in metrics_arg


def test_evaluate_model_handles_missing_data(tmp_path):
    """Test that evaluate_model handles missing data gracefully."""
    # Arrange
    model_path = str(tmp_path / "nonexistent_model")
    test_data_path = str(tmp_path / "nonexistent_data.parquet")

    # Act
    result = evaluate_model(
        model_path=model_path, test_data_path=test_data_path, execution_date="2023-01-03"
    )

    # Assert
    assert result is not None
    assert result["success"] is False
    assert "error" in result


@patch("src.models.evaluation.load_model")
def test_evaluate_model_handles_evaluation_errors(mock_load_model, sample_model_and_data):
    """Test that evaluate_model handles errors during evaluation."""
    # Arrange
    model_path = sample_model_and_data["model_path"]
    test_data_path = sample_model_and_data["test_data_path"]

    # Setup model to raise an error during prediction
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Prediction error")
    mock_model.config = MagicMock()
    mock_model.config.features = sample_model_and_data["feature_columns"]
    mock_load_model.return_value = mock_model

    # Act
    result = evaluate_model(
        model_path=model_path, test_data_path=test_data_path, execution_date="2023-01-03"
    )

    # Assert
    assert result is not None
    assert result["success"] is False
    assert "error" in result
    assert "Prediction error" in result["error"]
