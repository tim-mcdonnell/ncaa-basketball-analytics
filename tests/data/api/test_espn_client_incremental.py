import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, AsyncMock, MagicMock

from src.data.api.espn_client import AsyncESPNClient
from src.data.api.exceptions import APIError, ResourceNotFoundError

class TestESPNClientIncremental:
    """Test ESPN client incremental data features."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing metadata storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after test
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def metadata_file(self):
        """Return a standard metadata filename for tests."""
        return "test_metadata.json"
    
    @pytest.fixture
    def client(self, temp_dir, metadata_file):
        """Create an AsyncESPNClient for testing."""
        client = AsyncESPNClient(
            metadata_dir=temp_dir,
            metadata_file=metadata_file,
            timeout=1.0,  # Short timeout for tests
            rate_limit_initial=5,
            rate_limit_min=1,
            rate_limit_max=10
        )
        # Mock the _handle_response method to avoid actual HTTP requests
        client._handle_response = AsyncMock()
        # Mock the get method to avoid actual HTTP requests
        client.get = AsyncMock()
        # Mock the session to avoid context manager errors
        client.session = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_get_all_teams_incremental_no_cache(self, client):
        """Test getting all teams with incremental flag but no cached data."""
        # Mock the get method
        mock_response = {
            "sports": [
                {
                    "leagues": [
                        {
                            "teams": [
                                {"team": {"id": "1", "name": "Team 1"}},
                                {"team": {"id": "2", "name": "Team 2"}}
                            ]
                        }
                    ]
                }
            ]
        }
        
        client.get.return_value = mock_response
        
        # Call with incremental flag
        teams = await client.get_all_teams(incremental=True)
        
        # Should call API since no cached data
        client.get.assert_called_once_with("/teams")
        
        # Should return teams from API
        assert len(teams) == 2
        assert teams[0]["id"] == "1"
        assert teams[1]["id"] == "2"
        
        # Should update metadata
        metadata_path = os.path.join(client.metadata_dir, client.metadata_file)
        assert os.path.exists(metadata_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert "teams" in metadata
        assert "last_updated" in metadata["teams"]
    
    @pytest.mark.asyncio
    async def test_get_all_teams_incremental_with_cache(self, client, temp_dir, metadata_file):
        """Test getting all teams with incremental flag and cached data."""
        # Create mock metadata
        timestamp = "2023-01-01T12:00:00"
        test_metadata = {
            "teams": {
                "last_updated": timestamp
            }
        }
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, metadata_file), 'w') as f:
            json.dump(test_metadata, f)
        
        # Call with incremental flag
        teams = await client.get_all_teams(incremental=True)
        
        # Should not call API due to cached data
        client.get.assert_not_called()
        
        # Should return empty list (indicating no new data)
        assert teams == []
    
    @pytest.mark.asyncio
    async def test_get_team_updates_metadata(self, client):
        """Test that get_team updates metadata for specific team."""
        team_id = "123"
        team_data = {"name": "Test Team", "abbreviation": "TEST"}
        
        client.get.return_value = {"team": team_data}
        
        # Call get_team
        result = await client.get_team(team_id)
        
        # Should call API
        client.get.assert_called_once_with(f"/teams/{team_id}")
        
        # Should return team data
        assert result == team_data
        
        # Should update metadata
        metadata_path = os.path.join(client.metadata_dir, client.metadata_file)
        assert os.path.exists(metadata_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert "teams" in metadata
        assert "resources" in metadata["teams"]
        assert team_id in metadata["teams"]["resources"]
    
    @pytest.mark.asyncio
    async def test_get_games_incremental(self, client):
        """Test getting games with incremental flag."""
        date_str = "20230101"
        team_id = "123"
        games_data = {"events": [{"id": "game1"}, {"id": "game2"}]}
        
        client.get.return_value = games_data
        
        # Call get_games with incremental
        result = await client.get_games(
            date_str=date_str,
            team_id=team_id,
            incremental=True
        )
        
        # Should call API
        client.get.assert_called_once()
        
        # Should return games
        assert result == games_data["events"]
        
        # Should update metadata
        metadata_path = os.path.join(client.metadata_dir, client.metadata_file)
        assert os.path.exists(metadata_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert "games" in metadata
        assert "resources" in metadata["games"]
        
        # Resource ID should include date and team ID
        resource_id = f"{date_str}-{team_id}-all"
        assert resource_id in metadata["games"]["resources"]
    
    @pytest.mark.asyncio
    async def test_get_team_players_incremental(self, client):
        """Test getting team players with incremental updates."""
        team_id = "123"
        players_data = [{"id": "p1", "name": "Player 1"}, {"id": "p2", "name": "Player 2"}]
        
        client.get.return_value = {"team": {"athletes": players_data}}
        
        # Call get_team_players
        result = await client.get_team_players(team_id)
        
        # Should call API
        client.get.assert_called_once_with(f"/teams/{team_id}/roster")
        
        # Should return players
        assert result == players_data
        
        # Should update metadata
        metadata_path = os.path.join(client.metadata_dir, client.metadata_file)
        assert os.path.exists(metadata_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert "players" in metadata
        assert "resources" in metadata["players"]
        
        # Resource ID should include team ID and "players" suffix
        resource_id = f"{team_id}-players"
        assert resource_id in metadata["players"]["resources"] 