import os
import json
from typing import Dict, Any

from src.data.api.models.team import TeamResponse
from src.data.api.models.game import GameResponse
from src.data.api.models.player import PlayerResponse


class TestModelValidation:
    """Tests for model validation with inconsistent data."""

    def load_fixture(self, path: str) -> Dict[str, Any]:
        """Helper to load JSON fixture file."""
        fixture_path = os.path.join("tests", "data", "api", "fixtures", path)
        with open(fixture_path, "r") as f:
            return json.load(f)

    def test_team_model_with_record_object(self):
        """Test team model validates with record as object."""
        # Load fixture with record as object
        team_data = self.load_fixture("teams/team_detail.json")

        # Parse with model
        team = TeamResponse(**team_data["team"])

        # Assert
        assert team.id == "59"
        assert team.name == "Michigan"
        assert team.record == "15-10"  # Should extract from object

    def test_team_model_with_string_record(self):
        """Test team model validates with record as string."""
        # Load fixture with record as string
        team_data = self.load_fixture("teams/team_with_string_record.json")

        # Parse with model
        team = TeamResponse(**team_data["team"])

        # Assert
        assert team.id == "59"
        assert team.name == "Michigan"
        assert team.record == "15-10"  # Should use string directly

    def test_team_model_without_record(self):
        """Test team model validates without record field."""
        # Load fixture without record
        team_data = self.load_fixture("teams/team_without_record.json")

        # Parse with model
        team = TeamResponse(**team_data["team"])

        # Assert
        assert team.id == "59"
        assert team.name == "Michigan"
        assert team.record == "0-0"  # Should use default

    def test_game_model_in_progress(self):
        """Test game model validates with in-progress game."""
        # Load fixture for in-progress game
        game_data = self.load_fixture("games/game_in_progress.json")

        # Parse with model
        game = GameResponse(**game_data)

        # Assert
        assert game.id == "401516161"
        assert game.competitions[0].status.type.completed is False
        assert game.competitions[0].status.type.description == "In Progress"
        assert "2nd Half - 10:25" in game.competitions[0].status.type.detail

    def test_game_model_without_score(self):
        """Test game model validates without score field."""
        # Load fixture without score
        game_data = self.load_fixture("games/game_without_score.json")

        # Parse with model
        game = GameResponse(**game_data)

        # Assert
        assert game.id == "401516163"
        assert game.competitions[0].competitors[0].score is None  # Should handle missing score
        assert game.competitions[0].competitors[1].score is None

    def test_player_model_without_position(self):
        """Test player model validates without position field."""
        # Load fixture without position
        player_data = self.load_fixture("players/player_without_position.json")

        # Parse with model
        player = PlayerResponse(**player_data["player"])

        # Assert
        assert player.id == "101"
        assert player.fullName == "Player One"
        assert player.position is None  # Should handle missing position
