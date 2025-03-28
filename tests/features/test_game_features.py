"""
Test module for game features calculation functionality.
"""

import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from src.features.game_features import calculate_game_features

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
    """Create a mock DuckDB hook for testing."""
    mock_hook = MagicMock()

    # Mock games data
    games_df = pl.DataFrame(
        {
            "game_id": [f"GAME{i}" for i in range(1, 8)],
            "home_team": ["TEAM1", "TEAM2", "TEAM3", "TEAM1", "TEAM2", "TEAM3", "TEAM1"],
            "away_team": ["TEAM2", "TEAM3", "TEAM1", "TEAM3", "TEAM1", "TEAM2", "TEAM3"],
            "home_score": [70, 65, 80, 75, 90, 85, 65],
            "away_score": [65, 80, 75, 80, 85, 65, 70],
            "date": [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)
            ],
        }
    )

    # Mock team features data - specify dtype for feature_value to handle mixed data types
    team_features_df = pl.DataFrame(
        {
            "team_id": [
                "TEAM1",
                "TEAM1",
                "TEAM1",
                "TEAM2",
                "TEAM2",
                "TEAM2",
                "TEAM3",
                "TEAM3",
                "TEAM3",
            ],
            "feature_name": ["wins", "losses", "points_per_game"] * 3,
            "feature_value": pl.Series([5, 2, 75.5, 4, 3, 70.2, 3, 4, 72.8], dtype=pl.Float64),
            "feature_date": [datetime.now().strftime("%Y-%m-%d")] * 9,
            "lookback_days": [30] * 9,
        }
    )

    # Mock player features data - similarly specify dtype for feature_value
    player_features_df = pl.DataFrame(
        {
            "player_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "feature_name": ["points_per_game", "efficiency_rating"] * 3,
            "feature_value": pl.Series([18.5, 25.2, 12.8, 20.5, 15.6, 22.7], dtype=pl.Float64),
            "feature_date": [datetime.now().strftime("%Y-%m-%d")] * 6,
            "lookback_days": [30] * 6,
        }
    )

    # Mock hook methods
    mock_hook.get_polars_df.side_effect = lambda sql, parameters=None: {
        "SELECT * FROM games WHERE date >= ?": games_df,
        "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?": team_features_df,
        "SELECT * FROM player_features WHERE feature_date = ? AND lookback_days = ?": player_features_df,
    }.get(sql, pl.DataFrame())

    return mock_hook


def test_calculate_game_features_creates_correct_features(mock_duckdb_hook):
    """Test that calculate_game_features creates all required features."""
    # Arrange
    with patch("src.features.game_features.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook
        conn_id = "test_conn"
        database = "test_db"
        lookback_days = 30
        execution_date = datetime.now().strftime("%Y-%m-%d")

        # Act
        result = calculate_game_features(
            conn_id=conn_id,
            database=database,
            lookback_days=lookback_days,
            execution_date=execution_date,
        )

        # Assert
        assert result is not None

        # Verify connection was created with correct parameters
        MockDuckDBHook.assert_called_once_with(conn_id=conn_id, database=database)

        # Verify data was queried
        mock_duckdb_hook.get_polars_df.assert_any_call(
            "SELECT * FROM games WHERE date >= ?",
            {
                "1": (
                    datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=lookback_days)
                ).strftime("%Y-%m-%d")
            },
        )

        # Verify team features were queried
        mock_duckdb_hook.get_polars_df.assert_any_call(
            "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?",
            {"1": execution_date, "2": lookback_days},
        )

        # Verify feature data was written to the database
        assert mock_duckdb_hook.run_query.called
        # The first call should create the game_features table if it doesn't exist
        create_table_call = mock_duckdb_hook.run_query.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS game_features" in create_table_call[0][0]

        # Subsequent calls should insert feature values
        insert_calls = mock_duckdb_hook.run_query.call_args_list[1:]
        assert len(insert_calls) > 0
        for call in insert_calls:
            assert "INSERT OR REPLACE INTO game_features" in call[0][0]


def test_calculate_game_features_handles_no_games(mock_duckdb_hook):
    """Test that calculate_game_features handles the case where no games are found."""
    # Arrange
    empty_df = pl.DataFrame(
        {
            "game_id": [],
            "home_team": [],
            "away_team": [],
            "home_score": [],
            "away_score": [],
            "date": [],
        }
    )

    mock_duckdb_hook.get_polars_df.side_effect = lambda sql, parameters=None: {
        "SELECT * FROM games WHERE date >= ?": empty_df,
        "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?": pl.DataFrame(),
        "SELECT * FROM player_features WHERE feature_date = ? AND lookback_days = ?": pl.DataFrame(),
    }.get(sql, pl.DataFrame())

    with patch("src.features.game_features.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook

        # Act
        result = calculate_game_features(
            conn_id="test_conn",
            database="test_db",
            lookback_days=30,
            execution_date=datetime.now().strftime("%Y-%m-%d"),
        )

        # Assert
        assert result is not None
        # Should still create the game_features table
        assert mock_duckdb_hook.run_query.called
        create_table_call = mock_duckdb_hook.run_query.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS game_features" in create_table_call[0][0]

        # Should handle the empty data gracefully


def test_calculate_game_features_with_specific_execution_date(mock_duckdb_hook):
    """Test calculate_game_features with a specific execution date."""
    # Arrange
    specific_date = "2023-03-15"
    lookback_days = 30
    expected_query_date = (
        datetime.strptime(specific_date, "%Y-%m-%d") - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    with patch("src.features.game_features.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook

        # Act
        result = calculate_game_features(
            conn_id="test_conn",
            database="test_db",
            lookback_days=lookback_days,
            execution_date=specific_date,
        )

        # Assert
        assert result is not None

        # Verify correct date parameter was used in query
        mock_duckdb_hook.get_polars_df.assert_any_call(
            "SELECT * FROM games WHERE date >= ?", {"1": expected_query_date}
        )

        # Verify correct feature date was used for retrieving team features
        mock_duckdb_hook.get_polars_df.assert_any_call(
            "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?",
            {"1": specific_date, "2": lookback_days},
        )
