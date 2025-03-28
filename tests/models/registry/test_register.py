"""
Test module for model registry functionality.
"""

import pytest
import os
import json
from unittest.mock import MagicMock, patch

# Import the function to test (will be implemented)
from src.models.registry import register_model


@pytest.fixture
def sample_model_dir(tmp_path):
    """Create a sample model directory structure for testing."""
    # Create model directory
    model_dir = tmp_path / "models"
    os.makedirs(model_dir, exist_ok=True)

    # Create model config file
    model_config = {
        "model_type": "gradient_boosting",
        "features": ["win_ratio_diff", "points_per_game_diff", "efficiency_diff"],
        "hyperparameters": {"learning_rate": 0.05, "max_depth": 5},
        "training_params": {
            "training_date": "2023-01-01",
            "train_samples": 800,
            "val_samples": 200,
        },
        "version": {"major": 1, "minor": 0, "patch": 0},
    }

    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f)

    # Create empty model file
    with open(os.path.join(model_dir, "model.pt"), "w") as f:
        f.write("mock model content")

    return str(model_dir)


@patch("src.models.registry.register.MLflowClient")
def test_register_model_with_qualifying_metrics(mock_mlflow_client_class, sample_model_dir):
    """Test that register_model registers a model that meets quality criteria."""
    # Arrange
    mock_client = MagicMock()
    mock_mlflow_client_class.return_value = mock_client

    # Mock model registration response
    mock_client.create_model_version.return_value = MagicMock(version=1)

    # Act
    result = register_model(
        model_path=sample_model_dir,
        model_name="ncaa_basketball_prediction",
        tracking_uri="sqlite:///mlflow.db",
        min_accuracy=0.7,
        execution_date="2023-01-01",
        metrics={"accuracy": 0.75, "auc": 0.82},  # Good metrics
    )

    # Assert
    assert result is not None
    assert result["success"] is True
    assert "model_version" in result
    assert result["registered"] is True

    # Verify client was initialized with tracking URI
    mock_mlflow_client_class.assert_called_once_with(tracking_uri="sqlite:///mlflow.db")

    # Verify model was registered
    mock_client.create_registered_model.assert_called_once()
    mock_client.create_model_version.assert_called_once()


@patch("src.models.registry.register.MLflowClient")
def test_register_model_with_low_metrics(mock_mlflow_client_class, sample_model_dir):
    """Test that register_model doesn't register a model that fails to meet quality criteria."""
    # Arrange
    mock_client = MagicMock()
    mock_mlflow_client_class.return_value = mock_client

    # Act
    result = register_model(
        model_path=sample_model_dir,
        model_name="ncaa_basketball_prediction",
        tracking_uri="sqlite:///mlflow.db",
        min_accuracy=0.7,
        execution_date="2023-01-01",
        metrics={"accuracy": 0.65, "auc": 0.72},  # Low accuracy
    )

    # Assert
    assert result is not None
    assert result["success"] is True  # Overall operation succeeded
    assert result["registered"] is False  # But model wasn't registered
    assert "accuracy_threshold" in result
    assert result["accuracy_threshold"] == 0.7
    assert "actual_accuracy" in result
    assert result["actual_accuracy"] == 0.65

    # Verify model was not registered
    mock_client.create_model_version.assert_not_called()


@patch("src.models.registry.register.MLflowClient")
def test_register_model_with_no_metrics(mock_mlflow_client_class, sample_model_dir):
    """Test that register_model evaluates the model if no metrics are provided."""
    # Arrange
    mock_client = MagicMock()
    mock_mlflow_client_class.return_value = mock_client

    # Mock the evaluation function
    with patch("src.models.registry.register.evaluate_model") as mock_evaluate:
        mock_evaluate.return_value = {"success": True, "metrics": {"accuracy": 0.78, "auc": 0.85}}

        # Act
        result = register_model(
            model_path=sample_model_dir,
            model_name="ncaa_basketball_prediction",
            tracking_uri="sqlite:///mlflow.db",
            test_data_path="/path/to/test_data.parquet",  # Test data path provided
            min_accuracy=0.7,
            execution_date="2023-01-01",
        )

        # Assert
        assert result["success"] is True
        assert result["registered"] is True

        # Verify evaluate_model was called
        mock_evaluate.assert_called_once_with(
            model_path=sample_model_dir,
            test_data_path="/path/to/test_data.parquet",
            tracking_uri="sqlite:///mlflow.db",
            execution_date="2023-01-01",
        )


def test_register_model_handles_missing_model_files(tmp_path):
    """Test that register_model handles missing model files gracefully."""
    # Arrange
    nonexistent_path = str(tmp_path / "nonexistent")

    # Act
    result = register_model(
        model_path=nonexistent_path,
        model_name="ncaa_basketball_prediction",
        tracking_uri="sqlite:///mlflow.db",
        min_accuracy=0.7,
        execution_date="2023-01-01",
    )

    # Assert
    assert result is not None
    assert result["success"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()


@patch("src.models.registry.register.MLflowClient")
def test_register_model_handles_mlflow_errors(mock_mlflow_client_class, sample_model_dir):
    """Test that register_model handles errors in MLflow operations."""
    # Arrange
    mock_client = MagicMock()
    mock_client.create_model_version.side_effect = Exception("MLflow API error")
    mock_mlflow_client_class.return_value = mock_client

    # Act
    result = register_model(
        model_path=sample_model_dir,
        model_name="ncaa_basketball_prediction",
        tracking_uri="sqlite:///mlflow.db",
        min_accuracy=0.7,
        execution_date="2023-01-01",
        metrics={"accuracy": 0.75, "auc": 0.82},
    )

    # Assert
    assert result is not None
    assert result["success"] is False
    assert "error" in result
    assert "MLflow API error" in result["error"]
