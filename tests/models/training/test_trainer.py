import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import os

from src.models.training.trainer import ModelTrainer
from src.models.base import ModelConfig


class TestModelTrainer:
    """Test suite for the ModelTrainer class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple PyTorch model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self, input_size=3, hidden_size=5, output_size=1):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.activation = nn.ReLU()
                self.linear2 = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x
        
        return SimpleModel()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        # Create feature tensors
        X_train = torch.randn(100, 3)  # 100 samples, 3 features
        y_train = torch.randn(100, 1)  # 100 samples, 1 target
        
        X_val = torch.randn(20, 3)     # 20 samples, 3 features
        y_val = torch.randn(20, 1)     # 20 samples, 1 target
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val
        }
    
    def test_trainer_initialization(self, simple_model, sample_data):
        """Test initializing the model trainer."""
        # Setup
        train_loader = sample_data["train_loader"]
        val_loader = sample_data["val_loader"]
        
        # Define configuration
        config = ModelConfig(
            model_type="test_model",
            hyperparameters={
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "batch_size": 16
            },
            features=["feature1", "feature2", "feature3"],
            training_params={
                "num_epochs": 10,
                "early_stopping_patience": 3,
                "device": "cpu"
            }
        )
        
        # Initialize trainer
        trainer = ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        # Verify trainer is properly initialized
        assert trainer.model is simple_model, "Model not correctly assigned"
        assert trainer.train_loader is train_loader, "Train loader not correctly assigned"
        assert trainer.val_loader is val_loader, "Validation loader not correctly assigned"
        assert trainer.config is config, "Config not correctly assigned"
        assert isinstance(trainer.criterion, nn.Module), "Loss function not set properly"
        assert isinstance(trainer.optimizer, optim.Optimizer), "Optimizer not set properly"
    
    def test_training_loop(self, simple_model, sample_data):
        """Test that training correctly updates model parameters."""
        # Setup
        train_loader = sample_data["train_loader"]
        val_loader = sample_data["val_loader"]
        
        # Define configuration
        config = ModelConfig(
            model_type="test_model",
            hyperparameters={
                "learning_rate": 0.01,  # Higher lr for faster convergence in test
                "weight_decay": 0,
                "batch_size": 16
            },
            features=["feature1", "feature2", "feature3"],
            training_params={
                "num_epochs": 3,  # Few epochs for testing
                "early_stopping_patience": 5,
                "device": "cpu"
            }
        )
        
        # Save initial model weights
        initial_weights = {}
        for name, param in simple_model.named_parameters():
            initial_weights[name] = param.clone().detach()
        
        # Initialize trainer
        trainer = ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        # Train the model
        train_losses, val_losses = trainer.train()
        
        # Verify loss lists are returned
        assert len(train_losses) == 3, f"Expected 3 training losses, got {len(train_losses)}"
        assert len(val_losses) == 3, f"Expected 3 validation losses, got {len(val_losses)}"
        
        # Verify weights have been updated
        weights_changed = False
        for name, param in simple_model.named_parameters():
            if not torch.allclose(initial_weights[name], param):
                weights_changed = True
                break
        
        assert weights_changed, "Model weights did not change during training"
    
    def test_early_stopping(self, simple_model, sample_data, tmp_path):
        """Test that early stopping prevents overfitting."""
        # Setup
        train_loader = sample_data["train_loader"]
        val_loader = sample_data["val_loader"]
        
        # Create model save path
        model_dir = tmp_path / "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Define configuration with very low patience
        config = ModelConfig(
            model_type="test_model",
            hyperparameters={
                "learning_rate": 0.01,
                "weight_decay": 0,
                "batch_size": 16
            },
            features=["feature1", "feature2", "feature3"],
            training_params={
                "num_epochs": 20,  # Max epochs
                "early_stopping_patience": 2,  # Very low patience
                "device": "cpu",
                "model_dir": str(model_dir)
            }
        )
        
        # Create a mock validate method that returns increasing loss after a few epochs
        # to trigger early stopping
        original_validate = ModelTrainer.validate
        
        # Setup counter to track epochs
        epoch_counter = [0]
        
        def mock_validate(self):
            """Mock validation that returns increasing loss after first few epochs."""
            epoch_counter[0] += 1
            # Return decreasing loss for first 3 epochs, then increasing
            if epoch_counter[0] <= 3:
                loss = 1.0 - (0.1 * epoch_counter[0])
            else:
                loss = 0.7 + (0.1 * (epoch_counter[0] - 3))
            
            # Update metrics
            self.metrics.update_val_loss(loss)
            return loss
        
        # Patch the validate method
        ModelTrainer.validate = mock_validate
        
        try:
            # Initialize trainer
            trainer = ModelTrainer(
                model=simple_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config
            )
            
            # Train with patched validate method
            train_losses, val_losses = trainer.train()
            
            # Verify early stopping worked - should stop after validation loss increases for 2 epochs
            # (Initial decrease for 3 epochs, then increase for 2 more, so total 5 epochs)
            assert len(train_losses) <= 6, f"Early stopping failed, trained for {len(train_losses)} epochs"
            assert len(val_losses) <= 6, f"Early stopping failed, validated for {len(val_losses)} epochs"
        
        finally:
            # Restore original method
            ModelTrainer.validate = original_validate 