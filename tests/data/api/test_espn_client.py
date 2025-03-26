import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, ANY, call
from pathlib import Path

# Import the ESPN client that doesn't exist yet (will implement based on this test)
from src.data.api.espn_client import AsyncESPNClient


class TestAsyncESPNClient:
    """Tests for the ESPN specific API client."""
    
    def test_initialization(self):
        """Test ESPN client initializes with correct configuration."""
        # Arrange & Act
        client = AsyncESPNClient(timeout=30)
        
        # Assert
        assert client.base_url == "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        assert client.timeout == 30
    
    @pytest.mark.asyncio
    @patch("src.data.api.async_client.AsyncClient.get")
    async def test_get_teams(self, mock_get):
        """Test retrieving teams list."""
        # Arrange
        # Create a mock response based on real ESPN API structure
        mock_response = {
            "sports": [{
                "leagues": [{
                    "teams": [
                        {"team": {"id": "1", "name": "Team A", "abbreviation": "TA"}},
                        {"team": {"id": "2", "name": "Team B", "abbreviation": "TB"}}
                    ]
                }]
            }]
        }
        mock_get.return_value = mock_response
        
        client = AsyncESPNClient()
        
        # Act
        async with client:
            teams = await client.get_teams()
        
        # Assert
        # Use a more flexible assertion approach that doesn't depend on arg order
        assert mock_get.call_count == 1
        call_args = mock_get.call_args
        assert call_args[0][0] == "/teams"  # First positional arg
        assert len(teams) == 2
        assert teams[0]["id"] == "1"
        assert teams[0]["name"] == "Team A"
        assert teams[1]["id"] == "2"
        assert teams[1]["name"] == "Team B"
    
    @pytest.mark.asyncio
    @patch("src.data.api.async_client.AsyncClient.get")
    async def test_get_team_details(self, mock_get):
        """Test retrieving details for a specific team."""
        # Arrange
        team_id = "1"
        mock_response = {
            "team": {
                "id": "1",
                "name": "Team A",
                "abbreviation": "TA",
                "location": "Location A",
                "logo": "http://example.com/logo.png",
                "record": {"items": [{"summary": "10-5"}]}
            }
        }
        mock_get.return_value = mock_response
        
        client = AsyncESPNClient()
        
        # Act
        async with client:
            team = await client.get_team_details(team_id)
        
        # Assert
        assert mock_get.call_count == 1
        call_args = mock_get.call_args
        assert call_args[0][0] == f"/teams/{team_id}"  # First positional arg
        assert team["id"] == "1"
        assert team["name"] == "Team A"
        assert team["record"] == "10-5"
    
    @pytest.mark.asyncio
    @patch("src.data.api.async_client.AsyncClient.get")
    async def test_get_games(self, mock_get):
        """Test retrieving games for a date range."""
        # Arrange
        start_date = "20231101"
        end_date = "20231130"
        expected_params = {"dates": f"{start_date}-{end_date}", "limit": "100"}
        
        mock_response = {
            "events": [
                {
                    "id": "401", 
                    "date": "2023-11-15T19:00Z",
                    "name": "Team A vs Team B",
                    "competitions": [{
                        "competitors": [
                            {"team": {"id": "1"}, "score": "75"},
                            {"team": {"id": "2"}, "score": "70"}
                        ],
                        "status": {"type": {"completed": True}}
                    }]
                },
                {
                    "id": "402", 
                    "date": "2023-11-18T20:00Z",
                    "name": "Team C vs Team D",
                    "competitions": [{
                        "competitors": [
                            {"team": {"id": "3"}, "score": "80"},
                            {"team": {"id": "4"}, "score": "82"}
                        ],
                        "status": {"type": {"completed": True}}
                    }]
                }
            ]
        }
        mock_get.return_value = mock_response
        
        client = AsyncESPNClient()
        
        # Act
        async with client:
            games = await client.get_games(start_date=start_date, end_date=end_date)
        
        # Assert
        assert mock_get.call_count == 1
        call_args = mock_get.call_args
        assert call_args[0][0] == "/scoreboard"  # First positional arg
        
        # Check that params were passed (might be as positional or keyword arg)
        if len(call_args[0]) > 1:
            # Check positional arg
            assert call_args[0][1] == expected_params
        elif 'params' in call_args[1]:
            # Check keyword arg
            assert call_args[1]['params'] == expected_params
        else:
            pytest.fail("Params not found in call arguments")
        
        assert len(games) == 2
        assert games[0]["id"] == "401"
        assert games[0]["date"] == "2023-11-15T19:00Z"
        assert games[1]["id"] == "402"
        assert games[1]["date"] == "2023-11-18T20:00Z"
    
    @pytest.mark.asyncio
    @patch("src.data.api.async_client.AsyncClient.get")
    async def test_get_team_players(self, mock_get):
        """Test retrieving players for a specific team."""
        # Arrange
        team_id = "1"
        mock_response = {
            "team": {
                "athletes": [
                    {"id": "101", "fullName": "Player One", "position": {"name": "Guard"}},
                    {"id": "102", "fullName": "Player Two", "position": {"name": "Forward"}}
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = AsyncESPNClient()
        
        # Act
        async with client:
            players = await client.get_team_players(team_id)
        
        # Assert
        assert mock_get.call_count == 1
        call_args = mock_get.call_args
        assert call_args[0][0] == f"/teams/{team_id}/roster"  # First positional arg
        assert len(players) == 2
        assert players[0]["id"] == "101"
        assert players[0]["fullName"] == "Player One"  # Access the original fields since we don't transform in get_team_players
        assert players[1]["id"] == "102"
        assert players[1]["fullName"] == "Player Two" 