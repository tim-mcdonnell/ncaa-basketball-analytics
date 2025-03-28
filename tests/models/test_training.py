"""
Test module for model training functionality.
"""

import pytest
import os
import json
import polars as pl
from unittest.mock import MagicMock, patch, ANY

# Import the function to test (will be implemented)
from src.models.training import train_model


@pytest.fixture
def sample_training_data(tmp_path):
    """Create sample training data for testing."""
    # Create directories
    data_dir = tmp_path / "training_data"
    os.makedirs(data_dir, exist_ok=True)

    # Create feature columns
    feature_columns = ["win_percentage_diff", "points_per_game_diff", "predicted_point_diff"]

    # Save feature columns
    with open(os.path.join(data_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)

    # Create training data
    train_data = pl.DataFrame(
        {
            "game_id": [f"G{i}" for i in range(1, 21)],
            "home_team": [f"TEAM{i}" for i in range(1, 21)],
            "away_team": [f"TEAM{i+20}" for i in range(1, 21)],
            "date": ["2023-01-01"] * 20,
            "home_win": [True, False] * 10,
            "point_diff": [5.0, -3.0] * 10,
            "win_percentage_diff": [0.2, -0.1] * 10,
            "points_per_game_diff": [3.5, -2.8] * 10,
            "predicted_point_diff": [4.2, -3.1] * 10,
        }
    )

    # Create validation data (similar structure)
    val_data = pl.DataFrame(
        {
            "game_id": [f"G{i+20}" for i in range(1, 6)],
            "home_team": [f"TEAM{i+40}" for i in range(1, 6)],
            "away_team": [f"TEAM{i+45}" for i in range(1, 6)],
            "date": ["2023-01-02"] * 5,
            "home_win": [True, False, True, False, True],
            "point_diff": [6.0, -2.0, 4.5, -1.5, 3.0],
            "win_percentage_diff": [0.15, -0.05, 0.2, -0.1, 0.25],
            "points_per_game_diff": [2.5, -1.8, 3.0, -2.0, 4.0],
            "predicted_point_diff": [3.2, -2.1, 3.5, -1.8, 4.5],
        }
    )

    # Save data as parquet files
    train_data.write_parquet(os.path.join(data_dir, "train_data.parquet"))
    val_data.write_parquet(os.path.join(data_dir, "val_data.parquet"))

    return {
        "input_path": str(data_dir),
        "feature_columns": feature_columns,
        "train_data": train_data,
        "val_data": val_data,
    }


def test_train_model_creates_model_files(sample_training_data, tmp_path):
    """Test that train_model creates the expected model files."""
    # Arrange
    input_path = sample_training_data["input_path"]
    output_path = str(tmp_path / "models")
    model_type = "gradient_boosting"
    tracking_uri = "sqlite:///mlflow.db"
    execution_date = "2023-01-01"

    # Mock MLflow
    with patch("src.models.training.MLflowTracker") as MockMLflowTracker:
        mock_tracker = MagicMock()
        MockMLflowTracker.return_value = mock_tracker

        # Act
        result = train_model(
            input_path=input_path,
            output_path=output_path,
            model_type=model_type,
            tracking_uri=tracking_uri,
            execution_date=execution_date,
        )

        # Assert
        assert result is not None
        assert result.get("success") is True

        # Verify model files were created
        assert os.path.exists(os.path.join(output_path, "model.pt")), "Model file not created"
        assert os.path.exists(
            os.path.join(output_path, "model_config.json")
        ), "Model config not created"

        # Verify MLflow tracking was used
        MockMLflowTracker.assert_called_once_with(tracking_uri=tracking_uri)
        mock_tracker.start_run.assert_called_once()
        mock_tracker.log_params.assert_called()
        mock_tracker.log_metrics.assert_called()
        mock_tracker.log_artifacts.assert_called()


def test_train_model_handles_missing_data(tmp_path):
    """Test that train_model handles the case where training data is missing."""
    # Arrange
    input_path = str(tmp_path / "nonexistent_dir")
    output_path = str(tmp_path / "models")

    # Act
    result = train_model(
        input_path=input_path,
        output_path=output_path,
        model_type="gradient_boosting",
        tracking_uri="sqlite:///mlflow.db",
        execution_date="2023-01-01",
    )

    # Assert
    assert result is not None
    assert result.get("success") is False
    assert "error" in result


@patch("src.models.training.create_model")
def test_train_model_selects_correct_model_type(mock_create_model, sample_training_data, tmp_path):
    """Test that train_model creates the correct model type."""
    # Arrange
    input_path = sample_training_data["input_path"]
    output_path = str(tmp_path / "models")

    # Setup mock model
    mock_model = MagicMock()
    mock_model.train.return_value = {"train_loss": 0.1, "val_loss": 0.2}
    mock_create_model.return_value = mock_model

    # Act with different model types
    models_to_test = ["gradient_boosting", "neural_network", "logistic_regression"]

    for model_type in models_to_test:
        with patch("src.models.training.MLflowTracker"):
            result = train_model(
                input_path=input_path,
                output_path=output_path,
                model_type=model_type,
                execution_date="2023-01-01",
            )

            # Assert
            assert result.get("success") is True
            assert result.get("model_type") == model_type

            # Verify correct model type was created
            mock_create_model.assert_any_call(
                model_type=model_type, input_features=ANY, hyperparameters=ANY
            )


@patch("src.models.training.ModelTrainer")
def test_train_model_trains_model_correctly(mock_trainer_class, sample_training_data, tmp_path):
    """Test that train_model trains the model correctly."""
    # Arrange
    input_path = sample_training_data["input_path"]
    output_path = str(tmp_path / "models")

    # Setup mock trainer
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = {"train_loss": [0.5, 0.3, 0.2], "val_loss": [0.6, 0.4, 0.3]}
    mock_trainer_class.return_value = mock_trainer

    # Setup mock model
    mock_model = MagicMock()
    mock_model.save = MagicMock()

    # Patch other required components
    with (
        patch("src.models.training.create_model", return_value=mock_model),
        patch("src.models.training.MLflowTracker"),
    ):
        # Act
        result = train_model(
            input_path=input_path,
            output_path=output_path,
            model_type="gradient_boosting",
            execution_date="2023-01-01",
            hyperparameters={"learning_rate": 0.01, "max_depth": 3},
        )

        # Assert
        assert result.get("success") is True

        # Verify model was trained
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()

        # Verify model was saved
        mock_model.save.assert_called_once_with(os.path.join(output_path, "model.pt"))

        # Verify metrics were returned
        assert "metrics" in result
        assert "val_loss" in result["metrics"]
