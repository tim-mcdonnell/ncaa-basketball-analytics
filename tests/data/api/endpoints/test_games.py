import pytest
from unittest.mock import patch
import json
import os

from src.data.api.endpoints.games import get_games_by_date, get_game_details, get_games_by_team
from src.data.api.models.game import Game
from src.data.api.exceptions import ResourceNotFoundError


# Helper to load test fixtures
def load_fixture(filename):
    """Load a test fixture from the fixtures directory."""
    fixture_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", filename)
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture {fixture_path} not found")

    with open(fixture_path, "r") as f:
        return json.load(f)


class TestGamesEndpoint:
    """Tests for games endpoint functions."""

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_games")
    async def test_games_endpoint_get_games_by_date(self, mock_get_games):
        """Test retrieving games for a specific date works correctly."""
        # Arrange
        mock_games_data = [
            {
                "id": "401516161",
                "date": "2023-11-06T23:30Z",
                "name": "Team A vs Team B",
                "competitions": [
                    {
                        "competitors": [
                            {
                                "id": "1",
                                "homeAway": "home",
                                "team": {"id": "1", "name": "Team A"},
                                "score": "75",
                            },
                            {
                                "id": "2",
                                "homeAway": "away",
                                "team": {"id": "2", "name": "Team B"},
                                "score": "70",
                            },
                        ],
                        "status": {"type": {"completed": True}},
                    }
                ],
            },
            {
                "id": "401516162",
                "date": "2023-11-06T23:00Z",
                "name": "Team C vs Team D",
                "competitions": [
                    {
                        "competitors": [
                            {
                                "id": "3",
                                "homeAway": "home",
                                "team": {"id": "3", "name": "Team C"},
                                "score": "82",
                            },
                            {
                                "id": "4",
                                "homeAway": "away",
                                "team": {"id": "4", "name": "Team D"},
                                "score": "78",
                            },
                        ],
                        "status": {"type": {"completed": True}},
                    }
                ],
            },
        ]
        mock_get_games.return_value = mock_games_data

        test_date = "2023-11-06"

        # Act
        result = await get_games_by_date(test_date)

        # Assert
        assert len(result) == 2
        assert isinstance(result[0], Game)
        assert result[0].id == "401516161"
        assert result[0].home_team.team_id == "1"
        assert result[0].home_team.score == 75
        assert result[0].away_team.team_id == "2"
        assert result[0].away_team.score == 70
        assert result[0].status.completed is True

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_game")
    async def test_games_endpoint_get_game_details(self, mock_get_game):
        """Test retrieving detailed game information."""
        # Arrange
        mock_game_data = {
            "id": "401516161",
            "date": "2023-11-06T23:30Z",
            "name": "Michigan vs Michigan State",
            "competitions": [
                {
                    "competitors": [
                        {
                            "id": "1",
                            "homeAway": "home",
                            "team": {"id": "59", "name": "Michigan"},
                            "score": "75",
                        },
                        {
                            "id": "2",
                            "homeAway": "away",
                            "team": {"id": "127", "name": "Michigan State"},
                            "score": "70",
                        },
                    ],
                    "status": {"type": {"completed": True, "description": "Final"}},
                }
            ],
        }
        mock_get_game.return_value = mock_game_data

        # Act
        result = await get_game_details("401516161")

        # Assert
        assert isinstance(result, Game)
        assert result.id == "401516161"
        assert result.home_team.team_id == "59"
        assert result.home_team.team_name == "Michigan"
        assert result.home_team.score == 75
        assert result.home_team.is_home is True
        assert result.away_team.team_id == "127"
        assert result.away_team.team_name == "Michigan State"
        assert result.away_team.score == 70
        assert result.away_team.is_home is False
        assert result.status.completed is True

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_games")
    async def test_games_endpoint_get_games_by_team(self, mock_get_games):
        """Test filtering games by team works correctly."""
        # Arrange
        mock_games_data = [
            {
                "id": "401516161",
                "date": "2023-11-06T23:30Z",
                "name": "Michigan vs Michigan State",
                "competitions": [
                    {
                        "competitors": [
                            {
                                "id": "1",
                                "homeAway": "home",
                                "team": {"id": "59", "name": "Michigan"},
                                "score": "75",
                            },
                            {
                                "id": "2",
                                "homeAway": "away",
                                "team": {"id": "127", "name": "Michigan State"},
                                "score": "70",
                            },
                        ],
                        "status": {"type": {"completed": True}},
                    }
                ],
            },
            {
                "id": "401516162",
                "date": "2023-11-10T23:00Z",
                "name": "Michigan vs Ohio State",
                "competitions": [
                    {
                        "competitors": [
                            {
                                "id": "1",
                                "homeAway": "home",
                                "team": {"id": "59", "name": "Michigan"},
                                "score": "82",
                            },
                            {
                                "id": "2",
                                "homeAway": "away",
                                "team": {"id": "194", "name": "Ohio State"},
                                "score": "78",
                            },
                        ],
                        "status": {"type": {"completed": True}},
                    }
                ],
            },
        ]
        mock_get_games.return_value = mock_games_data

        # Act
        result = await get_games_by_team("59")

        # Assert
        assert len(result) == 2
        assert isinstance(result[0], Game)
        assert result[0].id == "401516161"
        assert result[0].home_team.team_id == "59"
        assert result[0].home_team.team_name == "Michigan"
        assert result[0].home_team.is_home is True
        assert result[1].id == "401516162"
        assert result[1].home_team.team_id == "59"
        assert result[1].home_team.team_name == "Michigan"
        assert result[1].home_team.is_home is True

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_game")
    async def test_games_endpoint_error_handling(self, mock_get_game):
        """Test error handling for nonexistent games."""
        # Arrange
        mock_get_game.side_effect = ResourceNotFoundError("Game", "999999")

        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await get_game_details("999999")
