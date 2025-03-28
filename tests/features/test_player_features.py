"""
Test module for player features calculation functionality.
"""

import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from src.features.player_features import calculate_player_features

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

    # Mock player data
    players_df = pl.DataFrame(
        {
            "player_id": ["P1", "P2", "P3"],
            "name": ["Player One", "Player Two", "Player Three"],
            "team_id": ["TEAM1", "TEAM1", "TEAM2"],
            "position": ["G", "F", "C"],
        }
    )

    # Mock player stats data
    player_stats_df = pl.DataFrame(
        {
            "player_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P1", "P2", "P3"],
            "game_id": [
                "GAME1",
                "GAME2",
                "GAME1",
                "GAME2",
                "GAME3",
                "GAME4",
                "GAME5",
                "GAME6",
                "GAME7",
            ],
            "minutes": [30, 32, 28, 25, 35, 33, 30, 25, 32],
            "points": [15, 18, 10, 14, 22, 20, 16, 12, 18],
            "rebounds": [3, 4, 8, 7, 12, 10, 2, 9, 11],
            "assists": [8, 7, 2, 3, 1, 2, 9, 1, 3],
            "steals": [2, 1, 1, 0, 1, 2, 3, 0, 1],
            "blocks": [0, 1, 1, 2, 3, 2, 0, 2, 3],
            "turnovers": [2, 3, 1, 2, 1, 2, 2, 1, 2],
            "date": [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 10)
            ],
        }
    )

    # Mock games data
    games_df = pl.DataFrame(
        {
            "game_id": [f"GAME{i}" for i in range(1, 8)],
            "home_team": ["TEAM1", "TEAM2", "TEAM3", "TEAM1", "TEAM2", "TEAM3", "TEAM1"],
            "away_team": ["TEAM2", "TEAM3", "TEAM1", "TEAM3", "TEAM1", "TEAM2", "TEAM3"],
            "date": [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)
            ],
        }
    )

    # Mock hook methods
    mock_hook.get_polars_df.side_effect = lambda sql, parameters=None: {
        "SELECT * FROM players": players_df,
        "SELECT ps.* FROM player_stats ps JOIN games g ON ps.game_id = g.game_id WHERE g.date >= ?": player_stats_df,
        "SELECT * FROM games WHERE date >= ?": games_df,
    }.get(sql, pl.DataFrame())

    return mock_hook


def test_calculate_player_features_creates_correct_features(mock_duckdb_hook):
    """Test that calculate_player_features creates all required features."""
    # Arrange
    with patch("src.features.player_features.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook
        conn_id = "test_conn"
        database = "test_db"
        lookback_days = 30
        execution_date = datetime.now().strftime("%Y-%m-%d")

        # Act
        result = calculate_player_features(
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
        mock_duckdb_hook.get_polars_df.assert_any_call("SELECT * FROM players")
        mock_duckdb_hook.get_polars_df.assert_any_call(
            "SELECT ps.* FROM player_stats ps JOIN games g ON ps.game_id = g.game_id WHERE g.date >= ?",
            {
                "1": (
                    datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=lookback_days)
                ).strftime("%Y-%m-%d")
            },
        )

        # Verify feature data was written to the database
        assert mock_duckdb_hook.run_query.called
        # The first call should create the player_features table if it doesn't exist
        create_table_call = mock_duckdb_hook.run_query.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS player_features" in create_table_call[0][0]

        # Subsequent calls should insert feature values
        insert_calls = mock_duckdb_hook.run_query.call_args_list[1:]
        assert len(insert_calls) > 0
        for call in insert_calls:
            assert "INSERT OR REPLACE INTO player_features" in call[0][0]


def test_calculate_player_features_handles_no_stats(mock_duckdb_hook):
    """Test that calculate_player_features handles the case where no player stats are found."""
    # Arrange
    empty_df = pl.DataFrame(
        {
            "player_id": [],
            "game_id": [],
            "minutes": [],
            "points": [],
            "rebounds": [],
            "assists": [],
            "steals": [],
            "blocks": [],
            "turnovers": [],
            "date": [],
        }
    )

    mock_duckdb_hook.get_polars_df.side_effect = lambda sql, parameters=None: {
        "SELECT * FROM players": pl.DataFrame(
            {"player_id": ["P1"], "name": ["Player One"], "team_id": ["TEAM1"]}
        ),
        "SELECT ps.* FROM player_stats ps JOIN games g ON ps.game_id = g.game_id WHERE g.date >= ?": empty_df,
    }.get(sql, pl.DataFrame())

    with patch("src.features.player_features.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook

        # Act
        result = calculate_player_features(
            conn_id="test_conn",
            database="test_db",
            lookback_days=30,
            execution_date=datetime.now().strftime("%Y-%m-%d"),
        )

        # Assert
        assert result is not None
        # Should still create the player_features table
        assert mock_duckdb_hook.run_query.called
        create_table_call = mock_duckdb_hook.run_query.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS player_features" in create_table_call[0][0]

        # Should handle the empty data gracefully


def test_calculate_player_features_with_specific_execution_date(mock_duckdb_hook):
    """Test calculate_player_features with a specific execution date."""
    # Arrange
    specific_date = "2023-03-15"
    lookback_days = 30
    expected_query_date = (
        datetime.strptime(specific_date, "%Y-%m-%d") - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    with patch("src.features.player_features.DuckDBHook") as MockDuckDBHook:
        MockDuckDBHook.return_value = mock_duckdb_hook

        # Act
        result = calculate_player_features(
            conn_id="test_conn",
            database="test_db",
            lookback_days=lookback_days,
            execution_date=specific_date,
        )

        # Assert
        assert result is not None

        # Verify correct date parameter was used in query
        mock_duckdb_hook.get_polars_df.assert_any_call(
            "SELECT ps.* FROM player_stats ps JOIN games g ON ps.game_id = g.game_id WHERE g.date >= ?",
            {"1": expected_query_date},
        )
