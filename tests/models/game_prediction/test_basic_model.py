import pytest
import torch
import torch.nn as nn
import os

from src.models.game_prediction.basic_model import BasicGamePredictionModel
from src.models.base import ModelConfig, ModelVersion


class TestBasicGamePredictionModel:
    """Test suite for BasicGamePredictionModel."""

    @pytest.fixture
    def model_config(self):
        """Create a model configuration for testing."""
        return ModelConfig(
            model_type="basic_game_prediction",
            hyperparameters={
                "input_size": 10,
                "hidden_size": 20,
                "output_size": 1,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
            },
            features=[f"feature_{i}" for i in range(10)],
            training_params={"num_epochs": 10, "early_stopping_patience": 3, "device": "cpu"},
        )

    def test_model_initialization(self, model_config):
        """Test that model initializes correctly with config."""
        # Initialize model
        model = BasicGamePredictionModel(model_config)

        # Verify model structure
        assert isinstance(model, nn.Module), "Model should be a PyTorch Module"
        assert hasattr(model, "network"), "Model should have a network attribute"
        assert isinstance(model.network, nn.Sequential), "Network should be a Sequential module"

        # Verify model layers
        layers = list(model.network.children())
        assert len(layers) > 2, "Model should have multiple layers"
        assert isinstance(layers[0], nn.Linear), "First layer should be Linear"

        # Verify model parameters
        input_size = model_config.hyperparameters["input_size"]
        hidden_size = model_config.hyperparameters["hidden_size"]
        output_size = model_config.hyperparameters["output_size"]

        first_layer = layers[0]
        assert (
            first_layer.in_features == input_size
        ), f"First layer input size should be {input_size}"
        assert (
            first_layer.out_features == hidden_size
        ), f"First layer output size should be {hidden_size}"

        # Find the last linear layer
        last_linear = None
        for layer in reversed(layers):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break

        assert last_linear is not None, "Could not find last linear layer"
        assert (
            last_linear.out_features == output_size
        ), f"Last layer output size should be {output_size}"

        # Verify config is stored
        assert model.config is model_config, "Model configuration not stored correctly"

        # Verify version is initialized
        assert hasattr(model, "version"), "Model should have a version attribute"
        assert isinstance(model.version, ModelVersion), "Version should be a ModelVersion instance"

    def test_forward_pass(self, model_config):
        """Test the forward pass of the model."""
        # Initialize model
        model = BasicGamePredictionModel(model_config)

        # Create a batch of data
        batch_size = 5
        input_size = model_config.hyperparameters["input_size"]
        inputs = torch.randn(batch_size, input_size)

        # Perform forward pass
        outputs = model(inputs)

        # Verify output shape
        output_size = model_config.hyperparameters["output_size"]
        expected_shape = (batch_size, output_size)
        assert (
            outputs.shape == expected_shape
        ), f"Expected output shape {expected_shape}, got {outputs.shape}"

    def test_predict_method(self, model_config):
        """Test the predict method."""
        # Initialize model
        model = BasicGamePredictionModel(model_config)

        # Create a batch of data
        batch_size = 5
        input_size = model_config.hyperparameters["input_size"]
        inputs = torch.randn(batch_size, input_size)

        # Get predictions
        predictions = model.predict(inputs)

        # Verify output shape and type
        output_size = model_config.hyperparameters["output_size"]
        expected_shape = (batch_size, output_size)

        assert isinstance(predictions, torch.Tensor), "Predictions should be a Tensor"
        assert (
            predictions.shape == expected_shape
        ), f"Expected prediction shape {expected_shape}, got {predictions.shape}"
        assert not predictions.requires_grad, "Predictions should not require gradients"

    def test_model_serialization(self, model_config, tmp_path):
        """Test saving and loading the model."""
        # Create save directory
        save_dir = tmp_path / "models"
        os.makedirs(save_dir, exist_ok=True)

        # Initialize model
        model = BasicGamePredictionModel(model_config)

        # Save model
        save_path = model.save(save_dir)

        # Verify files were created
        assert os.path.exists(save_path), f"Model file not created at {save_path}"

        # Create a new model and load
        loaded_model = BasicGamePredictionModel.load(save_path)

        # Verify model configuration was restored
        assert (
            loaded_model.config.model_type == model.config.model_type
        ), "Model type not restored correctly"
        assert (
            loaded_model.config.hyperparameters == model.config.hyperparameters
        ), "Hyperparameters not restored correctly"

        # Test if the weights are the same
        # (Compare predictions on same input)
        input_tensor = torch.randn(3, model_config.hyperparameters["input_size"])
        original_pred = model.predict(input_tensor)
        loaded_pred = loaded_model.predict(input_tensor)

        assert torch.allclose(
            original_pred, loaded_pred
        ), "Loaded model produces different predictions"

    def test_version_management(self, model_config):
        """Test model versioning."""
        # Initialize model
        model = BasicGamePredictionModel(model_config)

        # Get version info
        version = model.get_version()

        # Verify version attributes
        assert hasattr(version, "version_id"), "Version should have an ID"
        assert hasattr(version, "creation_timestamp"), "Version should have a timestamp"
        assert version.model_type == model_config.model_type, "Version has incorrect model type"
        assert (
            version.hyperparameters == model_config.hyperparameters
        ), "Version has incorrect hyperparameters"
        assert version.features == model_config.features, "Version has incorrect features"

    def test_get_hyperparameters(self, model_config):
        """Test retrieving hyperparameters."""
        # Initialize model
        model = BasicGamePredictionModel(model_config)

        # Get hyperparameters
        hyperparams = model.get_hyperparameters()

        # Verify hyperparameters
        assert (
            hyperparams == model_config.hyperparameters
        ), "get_hyperparameters() returned incorrect values"
