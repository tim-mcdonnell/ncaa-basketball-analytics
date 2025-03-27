import pytest
from unittest.mock import patch
import json
import os

from src.data.api.endpoints.teams import get_all_teams, get_team_details, get_teams_batch
from src.data.api.models.team import Team, TeamRecord
from src.data.api.exceptions import ResourceNotFoundError, APIError


# Helper to load test fixtures
def load_fixture(filename):
    """Load a test fixture from the fixtures directory."""
    fixture_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", filename)
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture {fixture_path} not found")

    with open(fixture_path, "r") as f:
        return json.load(f)


class TestTeamsEndpoint:
    """Tests for team endpoint functions."""

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_teams")
    async def test_teams_endpoint_get_all_teams(self, mock_get_teams):
        """Test retrieving all teams correctly formats the response."""
        # Arrange
        mock_teams_data = [
            {"id": "1", "name": "Team A", "abbreviation": "TA"},
            {"id": "2", "name": "Team B", "abbreviation": "TB"},
        ]
        mock_get_teams.return_value = mock_teams_data

        # Act
        result = await get_all_teams()

        # Assert
        assert len(result) == 2
        assert isinstance(result[0], Team)
        assert result[0].id == "1"
        assert result[0].name == "Team A"
        assert result[0].abbreviation == "TA"
        assert result[1].id == "2"
        assert result[1].name == "Team B"

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_team_details")
    async def test_teams_endpoint_get_team_details(self, mock_get_team_details):
        """Test retrieving detailed team information with record parsing."""
        # Arrange
        mock_team_data = {
            "id": "59",
            "name": "Michigan Wolverines",
            "abbreviation": "MICH",
            "location": "Ann Arbor",
            "logo": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/130.png",
            "record": "15-10",
        }
        mock_get_team_details.return_value = mock_team_data

        # Act
        result = await get_team_details("59")

        # Assert
        assert isinstance(result, Team)
        assert result.id == "59"
        assert result.name == "Michigan Wolverines"
        assert result.abbreviation == "MICH"
        assert result.location == "Ann Arbor"
        assert result.logo is not None
        assert result.record.summary == "15-10"
        assert result.record.wins == 15
        assert result.record.losses == 10

    @pytest.mark.asyncio
    @patch("src.data.api.endpoints.teams.get_team_details")
    async def test_teams_endpoint_get_teams_batch(self, mock_get_team_details):
        """Test concurrent team retrieval works correctly."""
        # Arrange
        team1 = Team(
            id="59",
            name="Michigan Wolverines",
            abbreviation="MICH",
            location="Ann Arbor",
            record=TeamRecord(summary="15-10", wins=15, losses=10),
        )
        team2 = Team(
            id="127",
            name="Michigan State Spartans",
            abbreviation="MSU",
            location="East Lansing",
            record=TeamRecord(summary="18-7", wins=18, losses=7),
        )

        # Configure the mock to return different values based on input
        async def side_effect(team_id, *args, **kwargs):
            if team_id == "59":
                return team1
            elif team_id == "127":
                return team2
            else:
                raise ResourceNotFoundError("Team", team_id)

        mock_get_team_details.side_effect = side_effect

        # Act
        result = await get_teams_batch(["59", "127", "999"])

        # Assert
        assert len(result) == 2
        assert result[0].id == "59"
        assert result[0].name == "Michigan Wolverines"
        assert result[1].id == "127"
        assert result[1].name == "Michigan State Spartans"

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.AsyncESPNClient.get_teams")
    async def test_teams_endpoint_api_error_handling(self, mock_get_teams):
        """Test error handling for API errors."""
        # Arrange
        mock_get_teams.side_effect = APIError("API request failed")

        # Act & Assert
        with pytest.raises(APIError):
            await get_all_teams()
