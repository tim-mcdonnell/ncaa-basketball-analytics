import pytest
from pydantic import ValidationError

from src.data.api.models.team import Team, TeamRecord, TeamList


class TestTeamModel:
    """Tests for Team model validation."""

    def test_team_model_validation_valid_data(self):
        """Test Team model with valid data."""
        # Arrange
        team_data = {
            "id": "59",
            "name": "Michigan Wolverines",
            "abbreviation": "MICH",
            "location": "Ann Arbor",
            "logo": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/130.png",
            "record": {"summary": "15-10", "wins": 15, "losses": 10},
        }

        # Act
        team = Team(**team_data)

        # Assert
        assert team.id == "59"
        assert team.name == "Michigan Wolverines"
        assert team.abbreviation == "MICH"
        assert team.location == "Ann Arbor"
        assert team.logo == "https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/130.png"
        assert team.record.summary == "15-10"
        assert team.record.wins == 15
        assert team.record.losses == 10

    def test_team_model_validation_minimal_data(self):
        """Test Team model with minimal required data."""
        # Arrange
        team_data = {"id": "59", "name": "Michigan Wolverines"}

        # Act
        team = Team(**team_data)

        # Assert
        assert team.id == "59"
        assert team.name == "Michigan Wolverines"
        assert team.abbreviation == ""
        assert team.location is None
        assert team.logo is None
        assert team.record.summary == "0-0"
        assert team.record.wins == 0
        assert team.record.losses == 0

    def test_team_model_validation_missing_required(self):
        """Test Team model fails with missing required fields."""
        # Arrange
        team_data = {"abbreviation": "MICH", "location": "Ann Arbor"}

        # Act & Assert
        with pytest.raises(ValidationError):
            Team(**team_data)

    def test_team_record_validation(self):
        """Test TeamRecord model validation."""
        # Arrange
        record_data = {"summary": "15-10", "wins": 15, "losses": 10}

        # Act
        record = TeamRecord(**record_data)

        # Assert
        assert record.summary == "15-10"
        assert record.wins == 15
        assert record.losses == 10

    def test_team_record_default_values(self):
        """Test TeamRecord default values."""
        # Act
        record = TeamRecord()

        # Assert
        assert record.summary == "0-0"
        assert record.wins == 0
        assert record.losses == 0

    def test_team_list_validation(self):
        """Test TeamList model validation."""
        # Arrange
        team1 = Team(id="59", name="Michigan Wolverines")
        team2 = Team(id="127", name="Michigan State Spartans")

        # Act
        team_list = TeamList(teams=[team1, team2])

        # Assert
        assert len(team_list.teams) == 2
        assert team_list.teams[0].id == "59"
        assert team_list.teams[1].id == "127"
