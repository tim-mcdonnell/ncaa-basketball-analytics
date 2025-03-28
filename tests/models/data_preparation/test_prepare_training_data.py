"""
Tests for the data preparation functions.
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from src.models.data_preparation.prepare_training_data import prepare_training_data

# Create mocks for airflow modules
airflow_mock = MagicMock()
sys.modules["airflow"] = airflow_mock
hooks_mock = MagicMock()
airflow_mock.hooks = hooks_mock
duckdb_hook_mock = MagicMock()
hooks_mock.duckdb_hook = duckdb_hook_mock
duckdb_hook_mock.DuckDBHook = MagicMock


@pytest.fixture
def mock_duckdb_hook():
    """Create a mock DuckDBHook."""
    mock_hook = MagicMock()

    # Mock game features data
    game_features_df = pl.DataFrame(
        {
            "game_id": [f"GAME{i}" for i in range(1, 11)],
            "home_team": [
                "TEAM1",
                "TEAM2",
                "TEAM3",
                "TEAM1",
                "TEAM2",
                "TEAM3",
                "TEAM1",
                "TEAM2",
                "TEAM3",
                "TEAM1",
            ],
            "away_team": [
                "TEAM2",
                "TEAM3",
                "TEAM1",
                "TEAM3",
                "TEAM1",
                "TEAM2",
                "TEAM3",
                "TEAM1",
                "TEAM2",
                "TEAM3",
            ],
            "feature_name": [
                "points_diff",
                "points_diff",
                "points_diff",
                "points_diff",
                "points_diff",
                "points_diff",
                "points_diff",
                "points_diff",
                "points_diff",
                "points_diff",
            ],
            "feature_value": [5, -15, 10, -5, 5, -10, 15, -5, 20, -10],
            "feature_date": [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 11)
            ],
            "result": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 1 for home win, 0 for away win
        }
    )

    # Mock games data with same games
    games_df = pl.DataFrame(
        {
            "game_id": [f"GAME{i}" for i in range(1, 11)],
            "home_team": [
                "TEAM1",
                "TEAM2",
                "TEAM3",
                "TEAM1",
                "TEAM2",
                "TEAM3",
                "TEAM1",
                "TEAM2",
                "TEAM3",
                "TEAM1",
            ],
            "away_team": [
                "TEAM2",
                "TEAM3",
                "TEAM1",
                "TEAM3",
                "TEAM1",
                "TEAM2",
                "TEAM3",
                "TEAM1",
                "TEAM2",
                "TEAM3",
            ],
            "home_score": [70, 65, 80, 75, 90, 85, 95, 70, 100, 75],
            "away_score": [65, 80, 70, 80, 85, 95, 80, 75, 80, 85],
            "date": [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 11)
            ],
            "season": [2023] * 10,
            "neutral_site": [False] * 10,
            "conference_game": [True] * 10,
            "tournament": [False] * 10,
        }
    )

    # Mock team features data
    team_features_df = pl.DataFrame(
        {
            "team_id": ["TEAM1", "TEAM2", "TEAM3"] * 10,  # 10 features per team
            "feature_name": [
                "wins",
                "losses",
                "points_per_game",
                "rebounds",
                "assists",
                "steals",
                "blocks",
                "turnovers",
                "fouls",
                "three_point_pct",
            ]
            * 3,
            "feature_value": [
                # TEAM1 features
                5,
                2,
                78.5,
                35.2,
                15.8,
                7.2,
                4.1,
                12.3,
                18.5,
                0.35,
                # TEAM2 features
                4,
                3,
                72.2,
                33.5,
                14.2,
                6.8,
                3.5,
                13.7,
                20.1,
                0.32,
                # TEAM3 features
                3,
                4,
                75.8,
                34.7,
                15.0,
                7.0,
                3.8,
                13.0,
                19.0,
                0.33,
            ],
            "feature_date": [datetime.now().strftime("%Y-%m-%d")] * 30,
            "lookback_days": [30] * 30,
        }
    )

    # Mock hook methods
    mock_hook.get_polars_df.side_effect = lambda sql, parameters=None: {
        "SELECT * FROM game_features WHERE feature_date >= ?": game_features_df,
        "SELECT * FROM games WHERE date >= ?": games_df,
        "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?": team_features_df,
    }.get(sql, pl.DataFrame())

    return mock_hook


def test_prepare_training_data_creates_correct_files(mock_duckdb_hook, tmp_path):
    """Test that prepare_training_data creates the expected files with correct data."""
    # Arrange
    with patch("src.models.data_preparation.prepare_data.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook
        conn_id = "test_conn"
        database = "test_db"
        lookback_days = 30
        execution_date = datetime.now().strftime("%Y-%m-%d")
        output_path = str(tmp_path)

        # Act
        result = prepare_training_data(
            conn_id=conn_id,
            database=database,
            lookback_days=lookback_days,
            execution_date=execution_date,
            output_path=output_path,
        )

        # Assert
        assert result is not None

        # Verify files were created
        assert os.path.exists(
            os.path.join(output_path, "train_data.parquet")
        ), "Training file not created"
        assert os.path.exists(
            os.path.join(output_path, "val_data.parquet")
        ), "Validation file not created"
        assert os.path.exists(
            os.path.join(output_path, "test_data.parquet")
        ), "Test file not created"
        assert os.path.exists(
            os.path.join(output_path, "feature_columns.json")
        ), "Feature columns file not created"

        # Verify connection was created with correct parameters
        MockDuckDBHook.assert_called_once_with(conn_id=conn_id, database=database)

        # Verify data was queried with correct parameters
        mock_duckdb_hook.get_polars_df.assert_any_call(
            "SELECT * FROM games WHERE date <= ? AND date >= ? AND status = 'final'",
            {
                "1": execution_date,
                "2": (
                    datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=lookback_days)
                ).strftime("%Y-%m-%d"),
            },
        )


def test_prepare_training_data_handles_no_games(mock_duckdb_hook, tmp_path):
    """Test that prepare_training_data handles the case where no games are found."""
    # Arrange
    # Return empty dataframes
    mock_duckdb_hook.get_polars_df.side_effect = lambda sql, parameters=None: pl.DataFrame()

    with patch("src.models.data_preparation.prepare_data.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook

        # Act
        result = prepare_training_data(
            conn_id="test_conn",
            database="test_db",
            lookback_days=30,
            execution_date=datetime.now().strftime("%Y-%m-%d"),
            output_path=str(tmp_path),
        )

        # Assert
        assert result is not None
        assert result.get("games_processed") == 0, "Should report zero games processed"
        assert result.get("success") is False, "Should report failure due to no data"


def test_prepare_training_data_splits_data_correctly(mock_duckdb_hook, tmp_path):
    """Test that prepare_training_data correctly splits data into train/val/test sets."""
    # Arrange
    with patch("src.models.data_preparation.prepare_data.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook
        output_path = str(tmp_path)

        # Act
        prepare_training_data(
            conn_id="test_conn",
            database="test_db",
            lookback_days=30,
            execution_date=datetime.now().strftime("%Y-%m-%d"),
            output_path=output_path,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        # Load the created files
        train_df = pl.read_parquet(os.path.join(output_path, "train_data.parquet"))
        val_df = pl.read_parquet(os.path.join(output_path, "val_data.parquet"))
        test_df = pl.read_parquet(os.path.join(output_path, "test_data.parquet"))

        # Verify split ratios (roughly, since we have a small sample)
        total_rows = len(train_df) + len(val_df) + len(test_df)
        assert total_rows > 0, "No data was written"
        assert len(train_df) / total_rows >= 0.5, "Train set should be approximately 60% of data"
        assert len(val_df) / total_rows <= 0.3, "Validation set should be approximately 20% of data"
        assert len(test_df) / total_rows <= 0.3, "Test set should be approximately 20% of data"

        # Verify no overlap between sets
        train_ids = set(train_df["game_id"].to_list())
        val_ids = set(val_df["game_id"].to_list())
        test_ids = set(test_df["game_id"].to_list())

        assert (
            len(train_ids.intersection(val_ids)) == 0
        ), "Train and validation sets should not overlap"
        assert len(train_ids.intersection(test_ids)) == 0, "Train and test sets should not overlap"
        assert (
            len(val_ids.intersection(test_ids)) == 0
        ), "Validation and test sets should not overlap"
