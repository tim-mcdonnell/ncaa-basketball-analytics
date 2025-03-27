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
            "name": "Player One",
            "jersey": "1",
            "position": "Guard",
            "team_id": "59",
            "height": "6'2\"",
            "weight": "185",
            "year": "Senior",
        }

        # Act
        player = Player(**player_data)

        # Assert
        assert player.id == "101"
        assert player.name == "Player One"
        assert player.jersey == "1"
        assert player.position == "Guard"
        assert player.team_id == "59"
        assert player.height == "6'2\""
        assert player.weight == "185"
        assert player.year == "Senior"

    def test_player_model_validation_minimal_data(self):
        """Test Player model with minimal required data."""
        # Arrange
        player_data = {"id": "101", "name": "Player One"}

        # Act
        player = Player(**player_data)

        # Assert
        assert player.id == "101"
        assert player.name == "Player One"
        assert player.jersey is None
        assert player.position is None
        assert player.team_id is None
        assert player.height is None
        assert player.weight is None
        assert player.year is None

    def test_player_model_validation_missing_required(self):
        """Test Player model fails with missing required fields."""
        # Arrange
        # Missing id
        player_data = {"name": "Player One", "jersey": "1", "position": "Guard"}

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
            name="Player One",
            jersey="1",
            position="Guard",
            team_id="59",
            height="6'2\"",
            weight="185",
            year="Senior",
        )

        # Assert - this test will need to be updated if any custom methods are added
        assert player.dict()["id"] == "101"
        assert player.dict()["name"] == "Player One"

    def test_player_model_optional_fields(self):
        """Test Player model handles optional fields correctly."""
        # Arrange & Act
        player = Player(id="101", name="Player One", jersey="1")

        # Act - update some fields
        player_dict = player.dict()
        player_dict["position"] = "Guard"
        updated_player = Player(**player_dict)

        # Assert
        assert updated_player.id == "101"
        assert updated_player.name == "Player One"
        assert updated_player.jersey == "1"
        assert updated_player.position == "Guard"
