import pytest
from pydantic import ValidationError
from datetime import datetime

from src.data.api.models.game import Game, GameStatus, TeamScore


class TestGameModel:
    """Tests for Game model validation."""

    def test_game_model_validation_valid_data(self):
        """Test Game model with valid data."""
        # Arrange
        game_data = {
            "id": "401516161",
            "date": "2023-11-06T23:30Z",
            "name": "Michigan vs Michigan State",
            "status": {"is_completed": True, "status_text": "Final"},
            "home_team": {"team_id": "59", "team_name": "Michigan", "score": 75},
            "away_team": {"team_id": "127", "team_name": "Michigan State", "score": 70},
        }

        # Act
        game = Game(**game_data)

        # Assert
        assert game.id == "401516161"
        assert game.name == "Michigan vs Michigan State"
        assert isinstance(game.date, datetime)
        assert game.status.is_completed is True
        assert game.status.description == "Final"
        assert game.home_team.team_id == "59"
        assert game.home_team.team_name == "Michigan"
        assert game.home_team.score == 75
        assert game.away_team.team_id == "127"
        assert game.away_team.team_name == "Michigan State"
        assert game.away_team.score == 70

    def test_game_model_validation_minimal_data(self):
        """Test Game model with minimal required data."""
        # Arrange
        game_data = {
            "id": "401516161",
            "date": "2023-11-06T23:30Z",
            "home_team": {"team_id": "59", "team_name": "Michigan"},
            "away_team": {"team_id": "127", "team_name": "Michigan State"},
        }

        # Act
        game = Game(**game_data)

        # Assert
        assert game.id == "401516161"
        assert isinstance(game.date, datetime)
        assert game.name is None
        assert game.status.completed is False
        assert game.status.description == ""
        assert game.home_team.team_id == "59"
        assert game.home_team.team_name == "Michigan"
        assert game.home_team.score == 0
        assert game.away_team.team_id == "127"
        assert game.away_team.team_name == "Michigan State"
        assert game.away_team.score == 0

    def test_game_model_validation_missing_required(self):
        """Test Game model fails with missing required fields."""
        # Arrange
        # Missing id and date
        game_data = {
            "home_team": {"team_id": "59", "team_name": "Michigan", "score": 75},
            "away_team": {"team_id": "127", "team_name": "Michigan State", "score": 70},
        }

        # Act & Assert
        with pytest.raises(ValidationError):
            Game(**game_data)

    def test_team_score_validation(self):
        """Test TeamScore model validation."""
        # Arrange
        team_score_data = {"team_id": "59", "team_name": "Michigan", "score": 75}

        # Act
        team_score = TeamScore(**team_score_data)

        # Assert
        assert team_score.team_id == "59"
        assert team_score.team_name == "Michigan"
        assert team_score.score == 75

    def test_team_score_default_values(self):
        """Test TeamScore default values."""
        # Arrange
        team_score_data = {"team_id": "59", "team_name": "Michigan"}

        # Act
        team_score = TeamScore(**team_score_data)

        # Assert
        assert team_score.team_id == "59"
        assert team_score.team_name == "Michigan"
        assert team_score.score == 0

    def test_game_status_validation(self):
        """Test GameStatus model validation."""
        # Arrange
        status_data = {"is_completed": True, "status_text": "Final"}

        # Act
        status = GameStatus(**status_data)

        # Assert
        assert status.is_completed is True
        assert status.status_text == "Final"

    def test_game_status_default_values(self):
        """Test GameStatus default values."""
        # Act
        status = GameStatus()

        # Assert
        assert status.completed is False
        assert status.description == ""
