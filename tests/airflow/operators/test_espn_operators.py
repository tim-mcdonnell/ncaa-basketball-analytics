from unittest.mock import patch

from airflow.operators.espn_operators import (
    FetchTeamsOperator,
    FetchGamesOperator,
    FetchPlayersOperator,
    FetchPlayerStatsOperator,
)


class TestESPNOperators:
    """Tests for ESPN API operators."""

    @patch("airflow.operators.espn_operators.DuckDBHook")
    @patch("airflow.operators.espn_operators.ESPNApiClient")
    def test_fetch_teams_operator(self, mock_api_client, mock_duckdb_hook):
        """Test that the FetchTeamsOperator correctly fetches and stores team data."""
        # Arrange
        mock_api_instance = mock_api_client.return_value
        mock_api_instance.get_teams.return_value = [
            {
                "team_id": "MICH",
                "name": "Michigan",
                "abbreviation": "MICH",
                "conference": "Big Ten",
            },
            {
                "team_id": "OSU",
                "name": "Ohio State",
                "abbreviation": "OSU",
                "conference": "Big Ten",
            },
        ]

        mock_hook_instance = mock_duckdb_hook.return_value

        # Act
        operator = FetchTeamsOperator(
            task_id="test_fetch_teams", conn_id="duckdb_default", database="test.duckdb"
        )
        operator.execute(context={})

        # Assert
        mock_api_client.assert_called_once()
        mock_api_instance.get_teams.assert_called_once()
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        # Verify data was stored
        mock_hook_instance.run_query.assert_called()  # Should have been called to store data

    @patch("airflow.operators.espn_operators.DuckDBHook")
    @patch("airflow.operators.espn_operators.ESPNApiClient")
    def test_fetch_games_operator(self, mock_api_client, mock_duckdb_hook):
        """Test that the FetchGamesOperator correctly fetches and stores game data."""
        # Arrange
        mock_api_instance = mock_api_client.return_value
        mock_api_instance.get_games.return_value = [
            {"game_id": "401236123", "home_team": "MICH", "away_team": "OSU", "date": "2023-01-01"},
            {"game_id": "401236124", "home_team": "IU", "away_team": "PUR", "date": "2023-01-02"},
        ]

        mock_hook_instance = mock_duckdb_hook.return_value

        # Act
        operator = FetchGamesOperator(
            task_id="test_fetch_games",
            conn_id="duckdb_default",
            database="test.duckdb",
            start_date="2023-01-01",
            end_date="2023-01-31",
        )
        operator.execute(context={})

        # Assert
        mock_api_client.assert_called_once()
        mock_api_instance.get_games.assert_called_once_with(
            start_date="2023-01-01", end_date="2023-01-31"
        )
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        # Verify data was stored
        mock_hook_instance.run_query.assert_called()  # Should have been called to store data

    @patch("airflow.operators.espn_operators.DuckDBHook")
    @patch("airflow.operators.espn_operators.ESPNApiClient")
    def test_fetch_players_operator(self, mock_api_client, mock_duckdb_hook):
        """Test that the FetchPlayersOperator correctly fetches and stores player data."""
        # Arrange
        mock_api_instance = mock_api_client.return_value
        mock_api_instance.get_players.return_value = [
            {"player_id": "4430185", "name": "Hunter Dickinson", "team_id": "MICH"},
            {"player_id": "4572153", "name": "Jett Howard", "team_id": "MICH"},
        ]

        mock_hook_instance = mock_duckdb_hook.return_value

        # Act
        operator = FetchPlayersOperator(
            task_id="test_fetch_players", conn_id="duckdb_default", database="test.duckdb"
        )
        operator.execute(context={})

        # Assert
        mock_api_client.assert_called_once()
        mock_api_instance.get_players.assert_called_once()
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        # Verify data was stored
        mock_hook_instance.run_query.assert_called()  # Should have been called to store data

    @patch("airflow.operators.espn_operators.DuckDBHook")
    @patch("airflow.operators.espn_operators.ESPNApiClient")
    def test_fetch_player_stats_operator(self, mock_api_client, mock_duckdb_hook):
        """Test that the FetchPlayerStatsOperator correctly fetches and stores player stats."""
        # Arrange
        mock_api_instance = mock_api_client.return_value
        mock_api_instance.get_player_stats.return_value = [
            {"player_id": "4430185", "game_id": "401236123", "points": 20, "rebounds": 10},
            {"player_id": "4572153", "game_id": "401236123", "points": 15, "rebounds": 5},
        ]

        mock_hook_instance = mock_duckdb_hook.return_value

        # Act
        operator = FetchPlayerStatsOperator(
            task_id="test_fetch_player_stats",
            conn_id="duckdb_default",
            database="test.duckdb",
            start_date="2023-01-01",
            end_date="2023-01-31",
        )
        operator.execute(context={})

        # Assert
        mock_api_client.assert_called_once()
        mock_api_instance.get_player_stats.assert_called_once_with(
            start_date="2023-01-01", end_date="2023-01-31"
        )
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")
        # Verify data was stored
        mock_hook_instance.run_query.assert_called()  # Should have been called to store data

    @patch("airflow.operators.espn_operators.DuckDBHook")
    @patch("airflow.operators.espn_operators.ESPNApiClient")
    def test_fetch_games_operator_with_incremental(self, mock_api_client, mock_duckdb_hook):
        """Test that the FetchGamesOperator correctly handles incremental data loading."""
        # Arrange
        mock_api_instance = mock_api_client.return_value
        mock_api_instance.get_games.return_value = [
            {"game_id": "401236123", "home_team": "MICH", "away_team": "OSU", "date": "2023-01-01"},
            {"game_id": "401236124", "home_team": "IU", "away_team": "PUR", "date": "2023-01-02"},
        ]

        mock_hook_instance = mock_duckdb_hook.return_value
        # Mock existing games to test incremental load
        mock_hook_instance.get_records.return_value = [("401236123",)]

        # Act
        operator = FetchGamesOperator(
            task_id="test_fetch_games_incremental",
            conn_id="duckdb_default",
            database="test.duckdb",
            start_date="2023-01-01",
            end_date="2023-01-31",
            incremental=True,
        )
        operator.execute(context={})

        # Assert
        mock_api_client.assert_called_once()
        mock_api_instance.get_games.assert_called_once_with(
            start_date="2023-01-01", end_date="2023-01-31"
        )
        mock_duckdb_hook.assert_called_once_with(conn_id="duckdb_default", database="test.duckdb")

        # Verify only new games were stored (401236124)
        for call_args in mock_hook_instance.run_query.call_args_list:
            query = call_args[0][0]
            if "INSERT" in query:
                assert "401236124" in query
                assert "401236123" not in query
