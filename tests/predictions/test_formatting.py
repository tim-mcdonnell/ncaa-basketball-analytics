"""
Test module for prediction formatting functionality.
"""

import pytest

import polars as pl

# Import the function to test (will be implemented)
from src.predictions.prediction import format_predictions


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    return pl.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "home_team": ["Duke", "Kentucky", "Gonzaga"],
            "away_team": ["North Carolina", "Kansas", "Baylor"],
            "win_probability": [0.75, 0.42, 0.83],
            "predicted_winner": ["Duke", "Kansas", "Gonzaga"],
            "prediction_date": ["2023-03-01", "2023-03-01", "2023-03-01"],
        }
    )


def test_format_predictions_default(sample_predictions):
    """Test formatting predictions with default parameters."""
    # Act
    result = format_predictions(sample_predictions)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 3

    # Check first game
    first_game = result[0]
    assert first_game["game_id"] == "G1"
    assert first_game["home_team"] == "Duke"
    assert first_game["away_team"] == "North Carolina"
    assert first_game["win_probability"] == 0.75
    assert first_game["win_probability_formatted"] == "75%"
    assert first_game["confidence_level"] == "High"
    assert first_game["predicted_winner"] == "Duke"
    assert first_game["prediction_date"] == "2023-03-01"


def test_format_predictions_with_confidence_thresholds(sample_predictions):
    """Test formatting predictions with custom confidence thresholds."""
    # Act
    result = format_predictions(
        sample_predictions, confidence_thresholds={"Low": 0.0, "Medium": 0.6, "High": 0.8}
    )

    # Assert
    assert result[0]["confidence_level"] == "Medium"  # 0.75 should be Medium with new thresholds
    assert result[1]["confidence_level"] == "Low"  # 0.42 should be Low
    assert result[2]["confidence_level"] == "High"  # 0.83 should be High


def test_format_predictions_include_only_high_confidence(sample_predictions):
    """Test formatting predictions with only high confidence predictions."""
    # Act
    result = format_predictions(sample_predictions, include_confidence=["High"])

    # Assert
    assert len(result) == 1  # Only one game with high confidence (default threshold)
    assert result[0]["game_id"] == "G3"  # Should be the Gonzaga game with 0.83 win probability


def test_format_predictions_with_empty_dataframe():
    """Test formatting predictions with an empty DataFrame."""
    # Arrange
    empty_predictions = pl.DataFrame(
        {
            "game_id": [],
            "home_team": [],
            "away_team": [],
            "win_probability": [],
            "predicted_winner": [],
            "prediction_date": [],
        }
    )

    # Act
    result = format_predictions(empty_predictions)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 0


def test_format_predictions_with_probability_formatting(sample_predictions):
    """Test different probability formatting options."""
    # Act - Test percentage formatting
    result_percentage = format_predictions(sample_predictions, probability_format="percentage")

    # Assert
    assert result_percentage[0]["win_probability_formatted"] == "75%"

    # Act - Test decimal formatting
    result_decimal = format_predictions(sample_predictions, probability_format="decimal")

    # Assert
    assert result_decimal[0]["win_probability_formatted"] == "0.75"

    # Act - Test fraction formatting
    result_fraction = format_predictions(sample_predictions, probability_format="fraction")

    # Assert
    assert result_fraction[0]["win_probability_formatted"] == "3/4"  # Approx 0.75


def test_format_predictions_custom_sort(sample_predictions):
    """Test different sorting options for predictions."""
    # Act - Sort by confidence descending
    result_confidence = format_predictions(
        sample_predictions, sort_by="confidence", sort_ascending=False
    )

    # Assert
    assert result_confidence[0]["game_id"] == "G3"  # Highest confidence (0.83)
    assert result_confidence[1]["game_id"] == "G1"  # Medium confidence (0.75)
    assert result_confidence[2]["game_id"] == "G2"  # Lowest confidence (0.42)

    # Act - Sort by game_id ascending
    result_game_id = format_predictions(sample_predictions, sort_by="game_id", sort_ascending=True)

    # Assert
    assert result_game_id[0]["game_id"] == "G1"
    assert result_game_id[1]["game_id"] == "G2"
    assert result_game_id[2]["game_id"] == "G3"
