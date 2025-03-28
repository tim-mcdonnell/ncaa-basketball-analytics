"""
Test module for prediction accuracy calculation functionality.
"""

import pytest
import polars as pl

from src.predictions.prediction import calculate_prediction_accuracy


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    return pl.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4", "G5"],
            "home_team": ["Duke", "Kentucky", "Gonzaga", "UCLA", "Villanova"],
            "away_team": ["North Carolina", "Kansas", "Baylor", "Arizona", "UConn"],
            "win_probability": [0.75, 0.42, 0.83, 0.68, 0.51],
            "predicted_winner": ["Duke", "Kansas", "Gonzaga", "UCLA", "Villanova"],
            "confidence_level": ["High", "Low", "High", "Medium", "Low"],
            "prediction_date": ["2023-03-01"] * 5,
        }
    )


@pytest.fixture
def sample_results():
    """Create sample actual results for testing."""
    return pl.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4", "G5"],
            "home_team": ["Duke", "Kentucky", "Gonzaga", "UCLA", "Villanova"],
            "away_team": ["North Carolina", "Kansas", "Baylor", "Arizona", "UConn"],
            "home_score": [78, 65, 85, 72, 68],
            "away_score": [70, 72, 78, 75, 71],
            "winning_team": ["Duke", "Kansas", "Gonzaga", "Arizona", "UConn"],
            "game_date": ["2023-03-01"] * 5,
        }
    )


def test_calculate_prediction_accuracy(sample_predictions, sample_results):
    """Test calculating prediction accuracy with sample data."""
    # Act
    accuracy_results = calculate_prediction_accuracy(sample_predictions, sample_results)

    # Assert
    assert "accuracy" in accuracy_results
    assert "games_evaluated" in accuracy_results
    assert "correct_predictions" in accuracy_results
    assert "accuracy_by_confidence" in accuracy_results

    # Overall accuracy: 3 out of 5 correct (Duke, Kansas, Gonzaga)
    assert accuracy_results["accuracy"] == 0.6
    assert accuracy_results["games_evaluated"] == 5
    assert accuracy_results["correct_predictions"] == 3

    # Accuracy by confidence level
    confidence_levels = accuracy_results["accuracy_by_confidence"]

    # High confidence: 2 out of 2 correct (Duke, Gonzaga)
    assert confidence_levels["High"]["accuracy"] == 1.0
    assert confidence_levels["High"]["games"] == 2
    assert confidence_levels["High"]["correct"] == 2

    # Medium confidence: 0 out of 1 correct (UCLA predicted, but Arizona won)
    assert confidence_levels["Medium"]["accuracy"] == 0.0
    assert confidence_levels["Medium"]["games"] == 1
    assert confidence_levels["Medium"]["correct"] == 0

    # Low confidence: 1 out of 2 correct (Kansas correct, Villanova incorrect)
    assert confidence_levels["Low"]["accuracy"] == 0.5
    assert confidence_levels["Low"]["games"] == 2
    assert confidence_levels["Low"]["correct"] == 1


def test_calculate_prediction_accuracy_no_matching_games():
    """Test calculating accuracy when no games match between predictions and results."""
    # Arrange
    predictions = pl.DataFrame(
        {"game_id": ["G1", "G2", "G3"], "predicted_winner": ["TeamA", "TeamB", "TeamC"]}
    )

    results = pl.DataFrame(
        {"game_id": ["G4", "G5", "G6"], "winning_team": ["TeamD", "TeamE", "TeamF"]}
    )

    # Act
    accuracy_results = calculate_prediction_accuracy(predictions, results)

    # Assert
    assert accuracy_results["accuracy"] == 0.0
    assert accuracy_results["games_evaluated"] == 0
    assert accuracy_results["correct_predictions"] == 0


def test_calculate_prediction_accuracy_with_subset_of_games():
    """Test calculating accuracy when only some games have results."""
    # Arrange
    predictions = pl.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4", "G5"],
            "predicted_winner": ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE"],
            "confidence_level": ["High", "Medium", "Low", "High", "Medium"],
        }
    )

    results = pl.DataFrame(
        {"game_id": ["G1", "G3", "G5"], "winning_team": ["TeamA", "TeamX", "TeamE"]}
    )

    # Act
    accuracy_results = calculate_prediction_accuracy(predictions, results)

    # Assert
    # 2 out of 3 correct (G1: TeamA correct, G3: TeamC incorrect, G5: TeamE correct)
    assert accuracy_results["accuracy"] == pytest.approx(2 / 3)
    assert accuracy_results["games_evaluated"] == 3
    assert accuracy_results["correct_predictions"] == 2

    # Check confidence level breakdown
    confidence_levels = accuracy_results["accuracy_by_confidence"]
    assert confidence_levels["High"]["accuracy"] == 1.0  # G1 correct
    assert confidence_levels["High"]["games"] == 1
    assert confidence_levels["Low"]["accuracy"] == 0.0  # G3 incorrect
    assert confidence_levels["Low"]["games"] == 1
    assert confidence_levels["Medium"]["accuracy"] == 1.0  # G5 correct
    assert confidence_levels["Medium"]["games"] == 1
