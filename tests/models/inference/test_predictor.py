import pytest
import torch
import polars as pl
import numpy as np
from unittest.mock import MagicMock

from src.models.inference.predictor import GamePredictor, batch_predict, create_feature_vector
from src.models.base import BaseModel


class TestFeatureProcessing:
    """Test suite for feature processing functions."""

    def test_create_feature_vector(self):
        """Test creating feature vectors from game data."""
        # Create mock game data
        game_data = pl.DataFrame(
            {
                "game_id": ["G1", "G2", "G3"],
                "home_team_id": ["TeamA", "TeamC", "TeamE"],
                "away_team_id": ["TeamB", "TeamD", "TeamF"],
                "game_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [0.5, 0.4, 0.3],
                "feature3": [0.7, 0.8, 0.9],
            }
        )

        # Define feature columns
        feature_columns = ["feature1", "feature2", "feature3"]

        # Create feature vectors
        feature_vectors = create_feature_vector(game_data, feature_columns)

        # Verify shape and type
        assert isinstance(feature_vectors, torch.Tensor), "Should return a tensor"
        assert feature_vectors.shape == (
            3,
            3,
        ), f"Expected shape (3, 3), got {feature_vectors.shape}"

        # Verify values
        expected_values = torch.tensor(
            [[0.1, 0.5, 0.7], [0.2, 0.4, 0.8], [0.3, 0.3, 0.9]], dtype=torch.float32
        )

        assert torch.allclose(feature_vectors, expected_values), "Feature values incorrect"


class TestBatchPrediction:
    """Test suite for batch prediction functions."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock(spec=BaseModel)

        # Define predict behavior
        def predict_side_effect(x):
            # Simple model that returns predictions based on sum of features
            return torch.sum(x, dim=1, keepdim=True) * 0.1

        model.predict = MagicMock(side_effect=predict_side_effect)

        # Set up features list
        model.config = MagicMock()
        model.config.features = ["feature1", "feature2", "feature3"]

        return model

    @pytest.fixture
    def sample_data(self):
        """Create sample game data for testing."""
        return pl.DataFrame(
            {
                "game_id": [f"G{i}" for i in range(1, 11)],
                "home_team_id": [f"Team{chr(65+i)}" for i in range(10)],
                "away_team_id": [f"Team{chr(75+i)}" for i in range(10)],
                "game_date": [f"2023-01-{i:02d}" for i in range(1, 11)],
                "feature1": np.random.rand(10).astype(np.float32),
                "feature2": np.random.rand(10).astype(np.float32),
                "feature3": np.random.rand(10).astype(np.float32),
            }
        )

    def test_batch_predict(self, mock_model, sample_data):
        """Test batch prediction functionality."""
        # Run batch predictions
        feature_columns = ["feature1", "feature2", "feature3"]
        predictions = batch_predict(mock_model, sample_data, feature_columns)

        # Verify predictions shape and type
        assert isinstance(predictions, torch.Tensor), "Should return a tensor"
        assert predictions.shape == (10, 1), f"Expected shape (10, 1), got {predictions.shape}"

        # Calculate expected predictions manually
        expected_values = sample_data.select(feature_columns).to_numpy()
        expected_preds = (
            torch.sum(torch.tensor(expected_values, dtype=torch.float32), dim=1, keepdim=True) * 0.1
        )

        # Verify predictions match expected values
        assert torch.allclose(predictions, expected_preds), "Predictions incorrect"
        # Verify the model's predict method was called
        mock_model.predict.assert_called_once()


class TestGamePredictor:
    """Test suite for the GamePredictor class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock(spec=BaseModel)

        # Define predict behavior
        def predict_side_effect(x):
            # Return win probability proportional to sum of features
            values = torch.sum(x, dim=1, keepdim=True) * 0.1
            return torch.clamp(values, 0.1, 0.9)  # Clamp to reasonable probabilities

        model.predict = MagicMock(side_effect=predict_side_effect)
        model.__call__ = MagicMock(side_effect=predict_side_effect)

        # Set up model configuration
        model.config = MagicMock()
        model.config.features = ["team_win_ratio", "point_differential", "efficiency"]
        model.config.model_type = "test_model"

        return model

    @pytest.fixture
    def sample_upcoming_games(self):
        """Create sample upcoming games data."""
        return pl.DataFrame(
            {
                "game_id": ["G1", "G2", "G3"],
                "game_date": ["2023-03-01", "2023-03-01", "2023-03-02"],
                "home_team_id": ["TeamA", "TeamC", "TeamE"],
                "home_team_name": ["Team A", "Team C", "Team E"],
                "away_team_id": ["TeamB", "TeamD", "TeamF"],
                "away_team_name": ["Team B", "Team D", "Team F"],
                "team_win_ratio": [0.7, 0.5, 0.3],
                "point_differential": [5.0, -2.0, 3.0],
                "efficiency": [1.1, 0.9, 1.0],
            }
        )

    def test_predictor_initialization(self, mock_model):
        """Test initializing the game predictor."""
        # Initialize predictor
        predictor = GamePredictor(model=mock_model)

        # Verify model assignment
        assert predictor.model is mock_model, "Model not assigned correctly"
        assert (
            predictor.feature_columns == mock_model.config.features
        ), "Feature columns not extracted from model"

    def test_predict_game(self, mock_model):
        """Test predicting a single game."""
        # Initialize predictor
        predictor = GamePredictor(model=mock_model)

        # Create game data
        game_data = {"team_win_ratio": 0.8, "point_differential": 7.5, "efficiency": 1.2}

        # Run prediction
        prediction = predictor.predict_game(game_data)

        # Calculate expected prediction
        features = torch.tensor([[0.8, 7.5, 1.2]], dtype=torch.float32)
        expected_prediction = torch.sum(features) * 0.1
        expected_prediction = torch.clamp(expected_prediction, 0.1, 0.9).item()

        # Verify prediction
        assert isinstance(prediction, float), "Should return a float"
        assert np.isclose(
            prediction, expected_prediction
        ), f"Expected {expected_prediction}, got {prediction}"

    def test_predict_games(self, mock_model, sample_upcoming_games):
        """Test predicting multiple games."""
        # Initialize predictor with point_spread prediction type
        predictor = GamePredictor(model=mock_model, prediction_type="point_spread")

        # Run predictions
        predictions_df = predictor.predict_games(sample_upcoming_games)

        # Verify dataframe structure
        assert isinstance(predictions_df, pl.DataFrame), "Should return a DataFrame"
        assert len(predictions_df) == 3, f"Expected 3 predictions, got {len(predictions_df)}"
        assert "predicted_point_spread" in predictions_df.columns, "Missing point spread column"
        assert "home_team_name" in predictions_df.columns, "Missing home team name"
        assert "away_team_name" in predictions_df.columns, "Missing away team name"

    def test_format_predictions(self, mock_model, sample_upcoming_games):
        """Test formatting prediction results."""
        # Initialize predictor
        predictor = GamePredictor(model=mock_model, prediction_type="win_probability")

        # Create raw predictions
        raw_predictions = torch.tensor([[0.75], [0.45], [0.60]])

        # Format predictions
        formatted = predictor.format_predictions(
            games_df=sample_upcoming_games, predictions=raw_predictions
        )

        # Verify dataframe structure
        assert "game_id" in formatted.columns, "Missing game ID"
        assert "game_date" in formatted.columns, "Missing game date"
        assert "win_probability" in formatted.columns, "Missing win probability"
        assert "predicted_winner" in formatted.columns, "Missing predicted winner"

        # Verify winner predictions
        winners = formatted["predicted_winner"].to_list()
        assert winners[0] == "Team A", "First game should predict Team A as winner"
        assert winners[1] == "Team D", "Second game should predict Team D as winner"
        assert winners[2] == "Team E", "Third game should predict Team E as winner"
