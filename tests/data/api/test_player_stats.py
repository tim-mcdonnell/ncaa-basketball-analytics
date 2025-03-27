import pytest
from unittest.mock import patch

from src.data.api.espn_client.client import AsyncESPNClient
from src.data.api.exceptions import ResourceNotFoundError


class TestPlayerStats:
    """Tests for the player statistics endpoint."""

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.client.AsyncESPNClient.get")
    async def test_get_player_stats(self, mock_get):
        """Test retrieving player statistics."""
        # Arrange
        player_id = "101"
        mock_response = {
            "player": {
                "id": "101",
                "fullName": "Player One",
                "statistics": {
                    "season": "2023-24",
                    "splits": {
                        "categories": [
                            {
                                "name": "scoring",
                                "stats": [
                                    {"name": "points", "value": 385},
                                    {"name": "ppg", "value": 15.4},
                                ],
                            }
                        ]
                    },
                },
            }
        }
        mock_get.return_value = mock_response
        client = AsyncESPNClient()

        # Act
        async with client:
            stats = await client.get_player_stats(player_id)

        # Assert
        assert mock_get.call_count == 1
        call_args = mock_get.call_args
        assert call_args[0][0] == f"/athletes/{player_id}/statistics"
        assert stats["player"]["id"] == "101"
        assert stats["player"]["fullName"] == "Player One"
        assert stats["player"]["statistics"]["season"] == "2023-24"
        assert stats["player"]["statistics"]["splits"]["categories"][0]["name"] == "scoring"

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.client.AsyncESPNClient.get_with_enhanced_recovery")
    async def test_get_player_stats_with_season(self, mock_get_recovery):
        """Test retrieving player statistics for a specific season."""
        # Arrange
        player_id = "101"
        season = "2023-24"
        mock_get_recovery.return_value = {"player": {"id": "101", "statistics": {"season": season}}}
        client = AsyncESPNClient()

        # Act
        async with client:
            stats = await client.get_player_stats(player_id, season=season)

        # Assert
        assert mock_get_recovery.call_count == 1
        call_args = mock_get_recovery.call_args
        assert call_args[0][0] == f"/athletes/{player_id}/statistics"
        assert call_args[0][1] == {"season": season}
        assert stats["player"]["id"] == "101"
        assert stats["player"]["statistics"]["season"] == season

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.client.AsyncESPNClient.get_with_enhanced_recovery")
    async def test_get_player_stats_not_found(self, mock_get_recovery):
        """Test handling when player statistics are not found."""
        # Arrange
        player_id = "999"  # Non-existent player
        mock_get_recovery.side_effect = ResourceNotFoundError("Player", player_id)
        client = AsyncESPNClient()

        # Act & Assert
        async with client:
            with pytest.raises(ResourceNotFoundError) as excinfo:
                await client.get_player_stats(player_id)

        assert "Player" in str(excinfo.value)
        assert player_id in str(excinfo.value)
        assert mock_get_recovery.call_count == 1
