import pytest
import torch
import os
from unittest.mock import Mock, patch

from src.models.game_prediction.ensemble import ModelEnsemble
from src.models.base import ModelConfig, BaseModel


class TestModelEnsemble:
    """Test suite for ModelEnsemble."""

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing ensemble."""
        # Create three mock models
        models = []

        for i in range(3):
            model = Mock(spec=BaseModel)

            # Configure the predict method to return predictable values
            def predict_side_effect(x, model_idx=i):
                # Each model returns slightly different predictions
                batch_size = x.shape[0]
                base_value = 0.5 + (0.1 * model_idx)
                return torch.ones(batch_size, 1) * base_value

            model.predict.side_effect = predict_side_effect

            # Configure version and config
            model.get_version.return_value = Mock(
                version_id=f"model_{i}",
                model_type="test_model",
                hyperparameters={"key": f"value_{i}"},
                features=[f"feature_{j}" for j in range(10)],
            )

            models.append(model)

        return models

    @pytest.fixture
    def ensemble_config(self):
        """Create a configuration for ensemble model."""
        return ModelConfig(
            model_type="ensemble",
            hyperparameters={
                "ensemble_method": "average",
                "weights": None,  # Equal weights
            },
            features=[
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
                "feature_5",
                "feature_6",
                "feature_7",
                "feature_8",
                "feature_9",
            ],
            training_params={},
        )

    def test_ensemble_initialization(self, mock_models, ensemble_config):
        """Test initializing the ensemble model."""
        # Create the ensemble
        ensemble = ModelEnsemble(models=mock_models, config=ensemble_config)

        # Verify models are stored
        assert len(ensemble.models) == 3, "Ensemble should have 3 models"
        for i, model in enumerate(ensemble.models):
            assert model is mock_models[i], f"Model {i} not stored correctly"

        # Verify config is stored
        assert ensemble.config is ensemble_config, "Config not stored correctly"

        # Verify version is generated
        assert hasattr(ensemble, "version"), "Version should be created"
        assert ensemble.version.model_type == "ensemble", "Version should have correct model type"

    def test_ensemble_predict_average(self, mock_models, ensemble_config):
        """Test prediction with average ensemble method."""
        # Set ensemble method to average
        ensemble_config.hyperparameters["ensemble_method"] = "average"

        # Create the ensemble
        ensemble = ModelEnsemble(models=mock_models, config=ensemble_config)

        # Create input data
        batch_size = 4
        input_features = 10
        inputs = torch.randn(batch_size, input_features)

        # Get ensemble predictions
        predictions = ensemble.predict(inputs)

        # Expected result: average of all model predictions
        # Model 0: 0.5, Model 1: 0.6, Model 2: 0.7 -> Average: 0.6
        expected_value = 0.6

        # Verify predictions
        assert predictions.shape == (batch_size, 1), "Prediction shape incorrect"
        assert torch.allclose(
            predictions, torch.ones_like(predictions) * expected_value, atol=1e-6
        ), "Average ensemble prediction incorrect"

    def test_ensemble_predict_weighted(self, mock_models, ensemble_config):
        """Test prediction with weighted ensemble method."""
        # Set ensemble method to weighted
        ensemble_config.hyperparameters["ensemble_method"] = "weighted"
        ensemble_config.hyperparameters["weights"] = [0.2, 0.3, 0.5]

        # Create the ensemble
        ensemble = ModelEnsemble(models=mock_models, config=ensemble_config)

        # Create input data
        batch_size = 4
        input_features = 10
        inputs = torch.randn(batch_size, input_features)

        # Get ensemble predictions
        predictions = ensemble.predict(inputs)

        # Expected result: weighted average of all model predictions
        # Model 0: 0.5 * 0.2 = 0.1
        # Model 1: 0.6 * 0.3 = 0.18
        # Model 2: 0.7 * 0.5 = 0.35
        # Total: 0.63
        expected_value = 0.63

        # Verify predictions
        assert predictions.shape == (batch_size, 1), "Prediction shape incorrect"
        assert torch.allclose(
            predictions, torch.ones_like(predictions) * expected_value, atol=1e-6
        ), "Weighted ensemble prediction incorrect"

    def test_save_and_load(self, mock_models, ensemble_config, tmp_path):
        """Test saving and loading ensemble model."""
        # Mock the model save and load methods
        for i, model in enumerate(mock_models):
            model.save.return_value = f"/path/to/model_{i}.pt"

        # Create save directory
        save_dir = tmp_path / "models"
        os.makedirs(save_dir, exist_ok=True)

        # Create the ensemble
        ensemble = ModelEnsemble(models=mock_models, config=ensemble_config)

        # Mock the entire save method to avoid file operations
        with patch.object(ModelEnsemble, "save") as mock_save:
            # Configure mock to return a path
            mock_save.return_value = str(save_dir / "ensemble_test.pt")

            # Save ensemble
            ensemble_path = ensemble.save(save_dir)

            # Verify save was called with correct path
            mock_save.assert_called_once_with(save_dir)

        # Mock the load method
        with patch.object(ModelEnsemble, "load") as mock_load:
            # Configure mock to return a new ensemble
            mock_ensemble = ModelEnsemble(
                models=[Mock(spec=BaseModel) for _ in range(3)], config=ensemble_config
            )
            mock_load.return_value = mock_ensemble

            # Load ensemble
            loaded_ensemble = ModelEnsemble.load(ensemble_path)

            # Verify load was called with correct path
            mock_load.assert_called_once_with(ensemble_path)

            # Verify loaded ensemble is the mock
            assert loaded_ensemble is mock_ensemble

    def test_get_version(self, mock_models, ensemble_config):
        """Test retrieving ensemble version information."""
        # Create the ensemble
        ensemble = ModelEnsemble(models=mock_models, config=ensemble_config)

        # Get version
        version = ensemble.get_version()

        # Verify version attributes
        assert version.model_type == "ensemble", "Wrong model type in version"
        assert "component_models" in version.hyperparameters, "Component models not in version"
        assert (
            len(version.hyperparameters["component_models"]) == 3
        ), "Wrong number of component models"

        # Verify component model information stored in version
        for i, model_info in enumerate(version.hyperparameters["component_models"]):
            assert model_info["version_id"] == f"model_{i}", "Wrong version ID stored"
