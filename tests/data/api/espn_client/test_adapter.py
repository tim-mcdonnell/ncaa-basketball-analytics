"""Tests for ESPNApiClient adapter."""

from unittest.mock import patch, MagicMock

from src.data.api.espn_client.adapter import ESPNApiClient


class TestESPNApiClient:
    """Tests for ESPNApiClient adapter."""

    @patch("src.data.api.espn_client.adapter.ESPNClient")
    def test_init(self, mock_espn_client):
        """Test initialization of ESPNApiClient."""
        # Arrange & Act
        client = ESPNApiClient()

        # Assert
        mock_espn_client.assert_called_once_with(config_path=None)
        assert client.client == mock_espn_client.return_value

    @patch("src.data.api.espn_client.adapter.ESPNClient")
    def test_get_teams(self, mock_espn_client):
        """Test get_teams method."""
        # Arrange
        mock_client = MagicMock()
        mock_espn_client.return_value = mock_client
        mock_client.get_teams.return_value = [
            {"team_id": "MICH", "name": "Michigan"},
            {"team_id": "OSU", "name": "Ohio State"},
        ]

        # Act
        client = ESPNApiClient()
        result = client.get_teams(season="2022-23")

        # Assert
        mock_client.get_teams.assert_called_once()
        assert len(result) == 2
        assert result[0]["team_id"] == "MICH"
        assert result[1]["team_id"] == "OSU"

    @patch("src.data.api.espn_client.adapter.ESPNClient")
    def test_get_games(self, mock_espn_client):
        """Test get_games method."""
        # Arrange
        mock_client = MagicMock()
        mock_espn_client.return_value = mock_client
        mock_client.get_games.return_value = [
            {"game_id": "401236123", "home_team": "MICH", "away_team": "OSU"},
            {"game_id": "401236124", "home_team": "IU", "away_team": "PUR"},
        ]

        # Act
        client = ESPNApiClient()
        result = client.get_games(
            start_date="2023-01-01",
            end_date="2023-01-31",
            team_id="MICH",
            limit=50,
        )

        # Assert
        mock_client.get_games.assert_called_once_with(
            start_date="2023-01-01",
            end_date="2023-01-31",
            team_id="MICH",
            limit=50,
        )
        assert len(result) == 2
        assert result[0]["game_id"] == "401236123"
        assert result[1]["game_id"] == "401236124"

    @patch("src.data.api.espn_client.adapter.ESPNClient")
    def test_get_players_with_team_id(self, mock_espn_client):
        """Test get_players method with team_id specified."""
        # Arrange
        mock_client = MagicMock()
        mock_espn_client.return_value = mock_client
        mock_client.get_team_players.return_value = [
            {"player_id": "4430185", "name": "Hunter Dickinson", "team_id": "MICH"},
            {"player_id": "4572153", "name": "Jett Howard", "team_id": "MICH"},
        ]

        # Act
        client = ESPNApiClient()
        result = client.get_players(team_id="MICH")

        # Assert
        mock_client.get_team_players.assert_called_once_with("MICH")
        assert len(result) == 2
        assert result[0]["player_id"] == "4430185"
        assert result[1]["player_id"] == "4572153"

    @patch("src.data.api.espn_client.adapter.ESPNClient")
    def test_get_players_all_teams(self, mock_espn_client):
        """Test get_players method without team_id specified."""
        # Arrange
        mock_client = MagicMock()
        mock_espn_client.return_value = mock_client

        # Setup teams
        mock_client.get_teams.return_value = [
            {"team_id": "MICH", "name": "Michigan"},
            {"team_id": "OSU", "name": "Ohio State"},
        ]

        # Setup players for each team
        mock_client.get_team_players.side_effect = [
            [
                {"player_id": "4430185", "name": "Hunter Dickinson", "team_id": "MICH"},
                {"player_id": "4572153", "name": "Jett Howard", "team_id": "MICH"},
            ],
            [
                {"player_id": "4432145", "name": "Justice Sueing", "team_id": "OSU"},
                {"player_id": "4433526", "name": "Zed Key", "team_id": "OSU"},
            ],
        ]

        # Act
        client = ESPNApiClient()
        result = client.get_players()

        # Assert
        assert mock_client.get_team_players.call_count == 2
        assert len(result) == 4
        assert result[0]["player_id"] == "4430185"
        assert result[2]["player_id"] == "4432145"

    @patch("src.data.api.espn_client.adapter.ESPNClient")
    def test_get_player_stats_with_player_id(self, mock_espn_client):
        """Test get_player_stats method with player_id specified."""
        # Arrange
        mock_client = MagicMock()
        mock_espn_client.return_value = mock_client
        mock_client.get_player_stats.return_value = {
            "player_id": "4430185",
            "name": "Hunter Dickinson",
            "stats": {"points": 18.2, "rebounds": 8.3},
        }

        # Act
        client = ESPNApiClient()
        result = client.get_player_stats(player_id="4430185")

        # Assert
        mock_client.get_player_stats.assert_called_once_with("4430185")
        assert len(result) == 1
        assert result[0]["player_id"] == "4430185"
        assert result[0]["stats"]["points"] == 18.2

    @patch("src.data.api.espn_client.adapter.ESPNClient")
    def test_get_player_stats_all_players(self, mock_espn_client):
        """Test get_player_stats method without player_id specified."""
        # Arrange
        mock_client = MagicMock()
        mock_espn_client.return_value = mock_client

        # Setup games
        mock_client.get_games.return_value = [
            {"game_id": "401236123", "home_team": "MICH", "away_team": "OSU"},
        ]

        # Setup players
        mock_client.get_teams.return_value = [{"team_id": "MICH"}]
        mock_client.get_team_players.return_value = [
            {"player_id": "4430185", "name": "Hunter Dickinson"},
            {"player_id": "4572153", "name": "Jett Howard"},
        ]

        # Setup player stats
        mock_client.get_player_stats.side_effect = [
            {"player_id": "4430185", "stats": {"points": 18.2}},
            {"player_id": "4572153", "stats": {"points": 14.5}},
        ]

        # Act
        client = ESPNApiClient()
        result = client.get_player_stats(start_date="2023-01-01", end_date="2023-01-31")

        # Assert
        assert mock_client.get_player_stats.call_count == 2
        assert len(result) == 2
        assert result[0]["player_id"] == "4430185"
        assert result[1]["player_id"] == "4572153"
