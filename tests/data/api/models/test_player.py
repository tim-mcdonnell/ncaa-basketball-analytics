import pytest
from pydantic import ValidationError

from src.data.api.models.player import Player


class TestPlayerModel:
    """Tests for Player model validation."""

    def test_player_model_validation_valid_data(self):
        """Test Player model with valid data."""
        # Arrange
        player_data = {
            "id": "101",
            "full_name": "Player One",
            "jersey": "1",
            "position": "Guard",
            "team_id": "59",
            "team_name": "Michigan",
            "headshot": "https://example.com/headshot.jpg",
        }

        # Act
        player = Player(**player_data)

        # Assert
        assert player.id == "101"
        assert player.full_name == "Player One"
        assert player.jersey == "1"
        assert player.position == "Guard"
        assert player.team_id == "59"
        assert player.team_name == "Michigan"
        assert player.headshot == "https://example.com/headshot.jpg"

    def test_player_model_validation_minimal_data(self):
        """Test Player model with minimal required data."""
        # Arrange
        player_data = {"id": "101", "full_name": "Player One"}

        # Act
        player = Player(**player_data)

        # Assert
        assert player.id == "101"
        assert player.full_name == "Player One"
        assert player.jersey is None
        assert player.position is None
        assert player.team_id is None
        assert player.team_name is None
        assert player.headshot is None

    def test_player_model_validation_missing_required(self):
        """Test Player model fails with missing required fields."""
        # Arrange
        # Missing id
        player_data = {"full_name": "Player One", "jersey": "1", "position": "Guard"}

        # Act & Assert
        with pytest.raises(ValidationError):
            Player(**player_data)

        # Missing name
        player_data = {"id": "101", "jersey": "1", "position": "Guard"}

        # Act & Assert
        with pytest.raises(ValidationError):
            Player(**player_data)

    def test_player_model_custom_methods(self):
        """Test any custom methods on the Player model."""
        # Arrange
        player = Player(
            id="101",
            full_name="Player One",
            jersey="1",
            position="Guard",
            team_id="59",
            team_name="Michigan",
            headshot="https://example.com/headshot.jpg",
        )

        # Assert - this test will need to be updated if any custom methods are added
        assert player.model_dump()["id"] == "101"
        assert player.model_dump()["full_name"] == "Player One"

    def test_player_model_optional_fields(self):
        """Test Player model handles optional fields correctly."""
        # Arrange & Act
        player = Player(id="101", full_name="Player One", jersey="1")

        # Act - update some fields
        updated_player = player.copy(update={"position": "Guard"})

        # Assert
        assert updated_player.id == "101"
        assert updated_player.full_name == "Player One"
        assert updated_player.jersey == "1"
        assert updated_player.position == "Guard"
