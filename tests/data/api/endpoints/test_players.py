import pytest
from unittest.mock import patch
import json
import os

from src.data.api.endpoints.players import get_players_by_team
from src.data.api.models.player import Player
from src.data.api.exceptions import APIError


# Helper to load test fixtures
def load_fixture(filename):
    """Load a test fixture from the fixtures directory."""
    fixture_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", filename)
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture {fixture_path} not found")

    with open(fixture_path, "r") as f:
        return json.load(f)


class TestPlayersEndpoint:
    """Tests for player endpoint functions."""

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_team_players")
    async def test_players_endpoint_get_team_roster(self, mock_get_team_players):
        """Test retrieving a team's roster works correctly."""
        # Arrange
        mock_players_data = [
            {
                "id": "101",
                "displayName": "Player One",
                "jersey": "1",
                "position": {"name": "Guard"},
            },
            {
                "id": "102",
                "displayName": "Player Two",
                "jersey": "2",
                "position": {"name": "Forward"},
            },
            {
                "id": "103",
                "displayName": "Player Three",
                "jersey": "3",
                "position": {"name": "Center"},
            },
        ]
        mock_get_team_players.return_value = mock_players_data

        # Act
        result = await get_players_by_team("59")

        # Assert
        assert len(result) == 3
        assert isinstance(result[0], Player)
        assert result[0].id == "101"
        assert result[0].full_name == "Player One"
        assert result[0].jersey == "1"
        assert result[0].position == "Guard"
        assert result[1].id == "102"
        assert result[1].full_name == "Player Two"
        assert result[2].id == "103"
        assert result[2].full_name == "Player Three"

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_team_players")
    async def test_players_endpoint_empty_roster(self, mock_get_team_players):
        """Test handling of empty roster."""
        # Arrange
        mock_get_team_players.return_value = []

        # Act
        result = await get_players_by_team("59")

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_team_players")
    async def test_players_endpoint_missing_fields(self, mock_get_team_players):
        """Test handling of players with missing fields."""
        # Arrange
        mock_players_data = [
            {"id": "101", "displayName": "Player One"},  # Missing position and jersey
            {
                "id": "102",
                "displayName": "Player Two",
                "position": {"name": "Forward"},
            },  # Missing jersey
            {"id": "103", "displayName": "Player Three", "jersey": "3"},  # Missing position
        ]
        mock_get_team_players.return_value = mock_players_data

        # Act
        result = await get_players_by_team("59")

        # Assert
        assert len(result) == 3
        assert result[0].id == "101"
        assert result[0].full_name == "Player One"
        assert result[0].jersey == ""
        assert result[0].position == ""
        assert result[1].id == "102"
        assert result[1].jersey == ""
        assert result[1].position == "Forward"
        assert result[2].id == "103"
        assert result[2].jersey == "3"
        assert result[2].position == ""

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_team_players")
    async def test_players_endpoint_error_handling(self, mock_get_team_players):
        """Test error handling in the players endpoint."""
        # Arrange
        mock_get_team_players.side_effect = APIError("Failed to retrieve players")

        # Act & Assert
        with pytest.raises(APIError):
            await get_players_by_team("59")
