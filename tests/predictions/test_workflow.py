"""
Test module for the prediction workflow functionality.
"""

import os
from unittest.mock import patch
import polars as pl

from src.predictions.main import run_prediction_workflow, evaluate_predictions


@patch("src.predictions.main.prepare_prediction_data")
@patch("src.predictions.main.generate_predictions")
@patch("src.predictions.main.format_predictions")
def test_run_prediction_workflow_successful(
    mock_format_predictions, mock_generate_predictions, mock_prepare_prediction_data, tmp_path
):
    """Test running a successful prediction workflow."""
    # Arrange
    database = "test.db"
    output_dir = str(tmp_path)

    # Mock the prepare_prediction_data function
    mock_prepare_prediction_data.return_value = {
        "success": True,
        "games_processed": 3,
        "output_files": {
            "prediction_data": "data/prediction_data.parquet",
            "feature_columns": "data/feature_columns.json",
        },
    }

    # Mock the generate_predictions function
    mock_generate_predictions.return_value = {
        "success": True,
        "games_predicted": 3,
        "output_file": "predictions/predictions.parquet",
    }

    # Mock the format_predictions function
    mock_format_predictions.return_value = [
        {"game_id": "G1", "home_team": "TeamA", "away_team": "TeamB", "win_probability": 0.75},
        {"game_id": "G2", "home_team": "TeamC", "away_team": "TeamD", "win_probability": 0.60},
        {"game_id": "G3", "home_team": "TeamE", "away_team": "TeamF", "win_probability": 0.82},
    ]

    # Create predictions directory and file
    predictions_dir = os.path.join(output_dir, "predictions", "2023-03-01")
    os.makedirs(predictions_dir, exist_ok=True)

    # Create mock predictions file
    mock_predictions = pl.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "home_team": ["TeamA", "TeamC", "TeamE"],
            "away_team": ["TeamB", "TeamD", "TeamF"],
            "win_probability": [0.75, 0.60, 0.82],
            "predicted_winner": ["TeamA", "TeamC", "TeamE"],
            "prediction_date": ["2023-03-01", "2023-03-01", "2023-03-01"],
        }
    )

    mock_predictions.write_parquet(os.path.join(predictions_dir, "predictions.parquet"))

    # Act
    result = run_prediction_workflow(
        database=database, prediction_date="2023-03-01", output_dir=output_dir
    )

    # Assert
    assert result["success"] is True
    assert result["workflow_completed"] is True
    assert result["games_predicted"] == 3
    assert result["formatted_predictions_count"] == 3
    assert "output_paths" in result

    # Verify function calls
    mock_prepare_prediction_data.assert_called_once()
    mock_generate_predictions.assert_called_once()
    mock_format_predictions.assert_called_once()

    # Verify output files
    formatted_file = os.path.join(predictions_dir, "formatted_predictions.json")
    assert os.path.exists(formatted_file)


@patch("src.predictions.main.prepare_prediction_data")
def test_run_prediction_workflow_with_no_games(mock_prepare_prediction_data, tmp_path):
    """Test running a prediction workflow when there are no games to predict."""
    # Arrange
    database = "test.db"
    output_dir = str(tmp_path)

    # Mock the prepare_prediction_data function with no games
    mock_prepare_prediction_data.return_value = {
        "success": True,
        "games_processed": 0,
        "output_files": {
            "prediction_data": "data/prediction_data.parquet",
            "feature_columns": "data/feature_columns.json",
        },
    }

    # Act
    result = run_prediction_workflow(
        database=database, prediction_date="2023-03-01", output_dir=output_dir
    )

    # Assert
    assert result["success"] is True
    assert result["workflow_completed"] is True
    assert result["games_predicted"] == 0
    assert "No games to predict" in result["message"]

    # Verify function calls
    mock_prepare_prediction_data.assert_called_once()


@patch("src.predictions.main.prepare_prediction_data")
def test_run_prediction_workflow_with_data_prep_error(mock_prepare_prediction_data, tmp_path):
    """Test running a prediction workflow when data preparation fails."""
    # Arrange
    database = "test.db"
    output_dir = str(tmp_path)

    # Mock the prepare_prediction_data function with an error
    mock_prepare_prediction_data.return_value = {
        "success": False,
        "error": "Failed to connect to database",
        "games_processed": 0,
    }

    # Act
    result = run_prediction_workflow(
        database=database, prediction_date="2023-03-01", output_dir=output_dir
    )

    # Assert
    assert result["success"] is False
    assert result["workflow_completed"] is False
    assert "Failed to connect to database" in result["error"]

    # Verify function calls
    mock_prepare_prediction_data.assert_called_once()


@patch("src.predictions.main.prepare_prediction_data")
@patch("src.predictions.main.generate_predictions")
def test_run_prediction_workflow_with_prediction_error(
    mock_generate_predictions, mock_prepare_prediction_data, tmp_path
):
    """Test running a prediction workflow when prediction generation fails."""
    # Arrange
    database = "test.db"
    output_dir = str(tmp_path)

    # Mock the prepare_prediction_data function
    mock_prepare_prediction_data.return_value = {
        "success": True,
        "games_processed": 3,
        "output_files": {
            "prediction_data": "data/prediction_data.parquet",
            "feature_columns": "data/feature_columns.json",
        },
    }

    # Mock the generate_predictions function with an error
    mock_generate_predictions.return_value = {
        "success": False,
        "error": "Model not found",
        "games_predicted": 0,
    }

    # Act
    result = run_prediction_workflow(
        database=database, prediction_date="2023-03-01", output_dir=output_dir
    )

    # Assert
    assert result["success"] is False
    assert result["workflow_completed"] is False
    assert "Model not found" in result["error"]
    assert "data_preparation" in result

    # Verify function calls
    mock_prepare_prediction_data.assert_called_once()
    mock_generate_predictions.assert_called_once()


@patch("src.predictions.main.calculate_prediction_accuracy")
def test_evaluate_predictions_successful(mock_calculate_accuracy, tmp_path):
    """Test evaluating predictions successfully."""
    # Arrange
    predictions_path = os.path.join(tmp_path, "predictions.parquet")
    results_path = os.path.join(tmp_path, "results.parquet")
    output_path = os.path.join(tmp_path, "accuracy.parquet")

    # Create mock prediction and results files
    predictions = pl.DataFrame(
        {"game_id": ["G1", "G2", "G3"], "predicted_winner": ["TeamA", "TeamC", "TeamE"]}
    )

    results = pl.DataFrame(
        {"game_id": ["G1", "G2", "G3"], "winning_team": ["TeamA", "TeamD", "TeamE"]}
    )

    predictions.write_parquet(predictions_path)
    results.write_parquet(results_path)

    # Mock accuracy calculation result
    mock_calculate_accuracy.return_value = {
        "accuracy": 0.67,
        "games_evaluated": 3,
        "correct_predictions": 2,
        "accuracy_by_confidence": {},
    }

    # Act
    result = evaluate_predictions(
        predictions_path=predictions_path, results_path=results_path, output_path=output_path
    )

    # Assert
    assert result["success"] is True
    assert "evaluation_results" in result
    assert result["evaluation_results"]["accuracy"] == 0.67
    assert result["evaluation_results"]["games_evaluated"] == 3
    assert result["evaluation_results"]["correct_predictions"] == 2

    # Verify output file was created
    assert os.path.exists(output_path)

    # Verify function call
    mock_calculate_accuracy.assert_called_once()


def test_evaluate_predictions_with_error(tmp_path):
    """Test evaluating predictions with file not found error."""
    # Arrange
    predictions_path = os.path.join(tmp_path, "nonexistent.parquet")
    results_path = os.path.join(tmp_path, "also_nonexistent.parquet")

    # Act
    result = evaluate_predictions(predictions_path=predictions_path, results_path=results_path)

    # Assert
    assert result["success"] is False
    assert "error" in result
