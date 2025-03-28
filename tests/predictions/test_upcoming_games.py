"""
Test module for upcoming games functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# Import the function to test (will be implemented)
from src.predictions.upcoming_games import fetch_upcoming_games


@pytest.fixture
def mock_duckdb_hook():
    """Create a mock DuckDB hook for testing."""
    mock_hook = MagicMock()

    # Mock existing games query
    mock_hook.get_records.return_value = [
        ("G1", "2023-03-01"),
        ("G2", "2023-03-02"),
    ]

    return mock_hook


@pytest.fixture
def mock_espn_api_client():
    """Create a mock ESPN API client for testing."""
    mock_client = MagicMock()

    # Create mock upcoming games
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    mock_games = [
        {
            "game_id": "G3",
            "home_team": "TEAM1",
            "away_team": "TEAM2",
            "date": today,
            "venue": "Stadium 1",
            "status": "scheduled",
        },
        {
            "game_id": "G4",
            "home_team": "TEAM3",
            "away_team": "TEAM4",
            "date": tomorrow,
            "venue": "Stadium 2",
            "status": "scheduled",
        },
        # Existing game that should be filtered out
        {
            "game_id": "G1",
            "home_team": "TEAM5",
            "away_team": "TEAM6",
            "date": today,
            "venue": "Stadium 3",
            "status": "scheduled",
        },
    ]

    mock_client.get_games.return_value = mock_games

    return mock_client


@patch("src.predictions.upcoming_games.DuckDBHook")
@patch("src.predictions.upcoming_games.ESPNApiClient")
def test_fetch_upcoming_games(
    mock_espn_api_client_class, mock_duckdb_hook_class, mock_espn_api_client, mock_duckdb_hook
):
    """Test fetching upcoming games."""
    # Arrange
    mock_espn_api_client_class.return_value = mock_espn_api_client
    mock_duckdb_hook_class.return_value = mock_duckdb_hook

    conn_id = "test_conn"
    database = "test_db"
    days_ahead = 2
    execution_date = datetime.now().strftime("%Y-%m-%d")

    # Act
    result = fetch_upcoming_games(
        conn_id=conn_id, database=database, days_ahead=days_ahead, execution_date=execution_date
    )

    # Assert
    assert result is not None
    assert result.get("success") is True
    assert result.get("games_fetched") == 3  # Total games from API
    assert result.get("new_games") == 2  # Only the new ones should be counted

    # Verify DuckDB hook was created with correct parameters
    mock_duckdb_hook_class.assert_called_once_with(conn_id=conn_id, database=database)

    # Verify ESPN API client was called with correct parameters
    mock_espn_api_client.get_games.assert_called_once()
    call_args = mock_espn_api_client.get_games.call_args[1]
    assert call_args["start_date"] == execution_date
    assert (
        datetime.strptime(call_args["end_date"], "%Y-%m-%d")
        - datetime.strptime(execution_date, "%Y-%m-%d")
    ).days == days_ahead

    # Verify games were inserted
    # Should call run_query twice - once for table creation, once for each insert
    assert mock_duckdb_hook.run_query.call_count >= 3


@patch("src.predictions.upcoming_games.DuckDBHook")
@patch("src.predictions.upcoming_games.ESPNApiClient")
def test_fetch_upcoming_games_with_no_new_games(mock_espn_api_client_class, mock_duckdb_hook_class):
    """Test fetching upcoming games when no new games are available."""
    # Arrange
    mock_espn_api_client = MagicMock()
    mock_espn_api_client.get_games.return_value = []
    mock_espn_api_client_class.return_value = mock_espn_api_client

    mock_duckdb_hook = MagicMock()
    mock_duckdb_hook_class.return_value = mock_duckdb_hook

    # Act
    result = fetch_upcoming_games(
        conn_id="test_conn",
        database="test_db",
        days_ahead=7,
        execution_date=datetime.now().strftime("%Y-%m-%d"),
    )

    # Assert
    assert result is not None
    assert result.get("success") is True
    assert result.get("games_fetched") == 0
    assert result.get("new_games") == 0


@patch("src.predictions.upcoming_games.DuckDBHook")
@patch("src.predictions.upcoming_games.ESPNApiClient")
def test_fetch_upcoming_games_handles_api_error(mock_espn_api_client_class, mock_duckdb_hook_class):
    """Test that fetch_upcoming_games handles API errors gracefully."""
    # Arrange
    mock_espn_api_client = MagicMock()
    mock_espn_api_client.get_games.side_effect = Exception("API Error")
    mock_espn_api_client_class.return_value = mock_espn_api_client

    mock_duckdb_hook = MagicMock()
    mock_duckdb_hook_class.return_value = mock_duckdb_hook

    # Act
    result = fetch_upcoming_games(
        conn_id="test_conn",
        database="test_db",
        days_ahead=7,
        execution_date=datetime.now().strftime("%Y-%m-%d"),
    )

    # Assert
    assert result is not None
    assert result.get("success") is False
    assert "error" in result
    assert "API Error" in result["error"]
