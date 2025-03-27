import pytest
import json
import os
from unittest.mock import patch, AsyncMock

from src.data.api.espn_client.client import AsyncESPNClient
from src.data.api.endpoints.teams import get_all_teams, get_team_details
from src.data.api.endpoints.games import get_game_details, get_games_by_team
from src.data.api.endpoints.players import get_players_by_team
from src.data.api.models.team import Team
from src.data.api.models.game import Game
from src.data.api.models.player import Player
from src.data.api.exceptions import APIError


# Helper to load test fixtures
def load_fixture(filename):
    """Load a test fixture from the fixtures directory."""
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", filename)
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture {fixture_path} not found")

    with open(fixture_path, "r") as f:
        return json.load(f)


class TestIntegrationDataFlow:
    """Tests for the complete data flow from API to validated models."""

    @pytest.mark.asyncio
    @patch("src.data.api.async_client.AsyncClient.get")
    async def test_end_to_end_team_data(self, mock_get):
        """Verify complete flow from API request to validated team models."""
        # Arrange
        team_list_response = {
            "sports": [
                {
                    "leagues": [
                        {
                            "teams": [
                                {
                                    "team": {
                                        "id": "59",
                                        "abbreviation": "MICH",
                                        "location": "Michigan",
                                        "name": "Wolverines",
                                    }
                                },
                                {
                                    "team": {
                                        "id": "127",
                                        "abbreviation": "MSU",
                                        "location": "Michigan State",
                                        "name": "Spartans",
                                    }
                                },
                            ]
                        }
                    ]
                }
            ]
        }

        team_detail_response = {
            "team": {
                "id": "59",
                "abbreviation": "MICH",
                "location": "Michigan",
                "name": "Wolverines",
                "logo": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/130.png",
                "record": {
                    "items": [
                        {
                            "summary": "15-10",
                            "stats": [
                                {"name": "wins", "value": 15},
                                {"name": "losses", "value": 10},
                            ],
                        }
                    ]
                },
            }
        }

        # Configure mock to return different responses based on URL
        async def get_side_effect(path, params=None):
            if path == "/teams":
                return team_list_response
            elif path == "/teams/59":
                return team_detail_response
            else:
                raise ValueError(f"Unexpected path: {path}")

        mock_get.side_effect = get_side_effect

        # Act - flow from client to model
        async with AsyncESPNClient() as client:
            # Get all teams
            teams = await get_all_teams(client)

            # Get details for a specific team
            michigan = await get_team_details("59", client)

        # Assert
        assert len(teams) == 2
        assert all(isinstance(team, Team) for team in teams)
        assert teams[0].id == "59"
        assert teams[0].abbreviation == "MICH"
        assert teams[0].location == "Michigan"
        assert teams[0].name == "Wolverines"

        assert isinstance(michigan, Team)
        assert michigan.id == "59"
        assert michigan.abbreviation == "MICH"
        assert michigan.location == "Michigan"
        assert michigan.name == "Wolverines"
        assert michigan.logo == "https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/130.png"
        assert michigan.record.summary == "15-10"
        assert michigan.record.wins == 15
        assert michigan.record.losses == 10

    @pytest.mark.asyncio
    @patch("src.data.api.async_client.AsyncClient.get")
    async def test_end_to_end_game_data(self, mock_get):
        """Verify complete flow from API request to validated game models."""
        # Arrange
        games_list = [
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
                        "status": {"type": {"completed": True, "description": "Final"}},
                        "venue": {"fullName": "Crisler Center"},
                        "attendance": "12500",
                    }
                ],
                "status": {"type": {"completed": True, "description": "Final"}},
            }
        ]

        game_data = {
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
                    "venue": {"fullName": "Crisler Center"},
                    "attendance": "12500",
                }
            ],
            "status": {"type": {"completed": True, "description": "Final"}},
        }

        # Configure mock to return different responses based on URL
        async def get_side_effect(path, params=None):
            if path == "/scoreboard":
                return {"events": games_list}
            elif path == "/competitions/401516161":
                return game_data
            else:
                raise ValueError(f"Unexpected path: {path}")

        mock_get.side_effect = get_side_effect

        # Add mock for get_games method
        with patch.object(AsyncESPNClient, "get_games", return_value=games_list):
            with patch.object(AsyncESPNClient, "get_game", return_value=game_data):
                # Act - flow from client to model
                async with AsyncESPNClient() as client:
                    # Get games for a team
                    games = await get_games_by_team("59", client)

                    # Get details for a specific game
                    game = await get_game_details("401516161", client)

        # Assert
        assert len(games) == 1
        assert all(isinstance(g, Game) for g in games)
        assert games[0].id == "401516161"

        assert isinstance(game, Game)
        assert game.id == "401516161"
        assert game.home_team.team_id == "59"
        assert game.home_team.team_name == "Michigan"
        assert game.home_team.score == 75
        assert game.home_team.is_home is True
        assert game.away_team.team_id == "127"
        assert game.away_team.team_name == "Michigan State"
        assert game.away_team.score == 70
        assert game.away_team.is_home is False
        assert game.status.completed is True
        assert game.venue == "Crisler Center"
        assert game.attendance == 12500

    @pytest.mark.asyncio
    @patch("src.data.api.async_client.AsyncClient.get")
    async def test_end_to_end_player_data(self, mock_get):
        """Verify complete flow from API request to validated player models."""
        # Arrange
        players_list = [
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
        ]

        roster_response = {"team": {"athletes": players_list}}

        # Configure mock to return roster response
        mock_get.return_value = roster_response

        # Add mock for get_team_players method
        with patch.object(AsyncESPNClient, "get_team_players", return_value=players_list):
            # Act - flow from client to model
            async with AsyncESPNClient() as client:
                # Get team roster
                players = await get_players_by_team("59", client)

        # Assert
        assert len(players) == 2
        assert all(isinstance(p, Player) for p in players)
        assert players[0].id == "101"
        assert players[0].full_name == "Player One"
        assert players[0].jersey == "1"
        assert players[0].position == "Guard"
        assert players[1].id == "102"
        assert players[1].full_name == "Player Two"

    @pytest.mark.asyncio
    @patch("src.data.api.async_client.AsyncClient.get")
    async def test_integration_error_handling(self, mock_get):
        """Test error handling throughout the integration flow."""
        # Arrange - simulate API error
        mock_get.side_effect = AsyncMock(side_effect=APIError("API request failed"))

        # Act & Assert - verify error propagation
        async with AsyncESPNClient() as client:
            with pytest.raises(APIError):
                await get_all_teams(client)
