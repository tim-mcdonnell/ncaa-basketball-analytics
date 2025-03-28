"""
Test module for prediction data preparation functionality.
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import polars as pl

# Import the function to test (will be implemented)
from src.predictions.data_preparation import prepare_prediction_data


@pytest.fixture
def mock_duckdb_hook():
    """Create a mock DuckDB hook for testing."""
    mock_hook = MagicMock()

    # Mock upcoming games data
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Mock games query result
    mock_games_df = pl.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "home_team": ["TEAM1", "TEAM3", "TEAM5"],
            "away_team": ["TEAM2", "TEAM4", "TEAM6"],
            "date": [today, tomorrow, tomorrow],
            "status": ["scheduled"] * 3,
            "venue": ["Venue 1", "Venue 2", "Venue 3"],
        }
    )

    # Mock team features query result
    mock_team_features_df = pl.DataFrame(
        {
            "team_id": ["TEAM1", "TEAM1", "TEAM2", "TEAM2", "TEAM3", "TEAM4"],
            "feature_date": [today] * 6,
            "feature_name": ["win_percentage", "points_per_game"] * 3,
            "feature_value": [0.75, 78.2, 0.65, 72.1, 0.80, 85.3],
            "lookback_days": [30] * 6,
        }
    )

    # Configure mock get_polars_df to return different data based on query
    mock_hook.get_polars_df.side_effect = lambda sql, params=None: {
        "SELECT * FROM games WHERE status = 'scheduled' AND date >= ?": mock_games_df,
        "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?": mock_team_features_df,
    }.get(sql, pl.DataFrame())

    return mock_hook


@patch("src.predictions.data_preparation.DuckDBHook")
def test_prepare_prediction_data_creates_correct_files(
    mock_duckdb_hook_class, mock_duckdb_hook, tmp_path
):
    """Test that prepare_prediction_data creates the expected files with correct data."""
    # Arrange
    mock_duckdb_hook_class.return_value = mock_duckdb_hook

    conn_id = "test_conn"
    database = "test_db"
    lookback_days = 30
    execution_date = datetime.now().strftime("%Y-%m-%d")
    output_path = str(tmp_path)

    # Act
    result = prepare_prediction_data(
        conn_id=conn_id,
        database=database,
        lookback_days=lookback_days,
        execution_date=execution_date,
        output_path=output_path,
    )

    # Assert
    assert result is not None
    assert result["success"] is True
    assert result["games_processed"] == 3  # All three games from the mock data

    # Verify files were created
    assert os.path.exists(
        os.path.join(output_path, "prediction_data.parquet")
    ), "Prediction data file not created"
    assert os.path.exists(
        os.path.join(output_path, "feature_columns.json")
    ), "Feature columns file not created"

    # Load and verify data
    prediction_data = pl.read_parquet(os.path.join(output_path, "prediction_data.parquet"))
    assert len(prediction_data) == 3, f"Expected 3 games, got {len(prediction_data)}"
    assert "game_id" in prediction_data.columns
    assert "home_team" in prediction_data.columns
    assert "away_team" in prediction_data.columns
    assert "date" in prediction_data.columns

    # Verify feature columns were included
    for feature_col in ["win_percentage_diff", "points_per_game_diff"]:
        assert feature_col in prediction_data.columns, f"Missing feature column: {feature_col}"


@patch("src.predictions.data_preparation.DuckDBHook")
def test_prepare_prediction_data_handles_no_games(mock_duckdb_hook_class, tmp_path):
    """Test that prepare_prediction_data handles the case where no games are found."""
    # Arrange
    mock_hook = MagicMock()
    mock_hook.get_polars_df.return_value = pl.DataFrame()  # Empty dataframe for all queries
    mock_duckdb_hook_class.return_value = mock_hook

    # Act
    result = prepare_prediction_data(
        conn_id="test_conn",
        database="test_db",
        lookback_days=30,
        execution_date=datetime.now().strftime("%Y-%m-%d"),
        output_path=str(tmp_path),
    )

    # Assert
    assert result is not None
    assert result["success"] is True
    assert result["games_processed"] == 0, "Should report zero games processed"


@patch("src.predictions.data_preparation.DuckDBHook")
def test_prepare_prediction_data_handles_missing_features(mock_duckdb_hook_class, tmp_path):
    """Test that prepare_prediction_data handles the case where team features are missing."""
    # Arrange
    mock_hook = MagicMock()

    # Mock games data but no team features
    mock_games_df = pl.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "home_team": ["TEAM1", "TEAM3"],
            "away_team": ["TEAM2", "TEAM4"],
            "date": [datetime.now().strftime("%Y-%m-%d")] * 2,
            "status": ["scheduled"] * 2,
        }
    )

    # Configure mock to return games but no features
    mock_hook.get_polars_df.side_effect = lambda sql, params=None: {
        "SELECT * FROM games WHERE status = 'scheduled' AND date >= ?": mock_games_df,
        "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?": pl.DataFrame(),
    }.get(sql, pl.DataFrame())

    mock_duckdb_hook_class.return_value = mock_hook

    # Act
    result = prepare_prediction_data(
        conn_id="test_conn",
        database="test_db",
        lookback_days=30,
        execution_date=datetime.now().strftime("%Y-%m-%d"),
        output_path=str(tmp_path),
    )

    # Assert
    assert result is not None
    assert result["success"] is True
    assert result["games_processed"] == 2  # Should still process games

    # Check that output files exist
    assert os.path.exists(os.path.join(str(tmp_path), "prediction_data.parquet"))

    # Load data and check that it contains minimal info
    prediction_data = pl.read_parquet(os.path.join(str(tmp_path), "prediction_data.parquet"))
    assert len(prediction_data) == 2
    assert "game_id" in prediction_data.columns
    assert "home_team" in prediction_data.columns
    assert "away_team" in prediction_data.columns
