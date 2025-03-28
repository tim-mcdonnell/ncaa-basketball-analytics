import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import os

from ..base import BaseModel, ModelConfig, ModelVersion


class BasicGamePredictionModel(BaseModel):
    """
    Basic neural network model for game prediction.
    
    This model uses a simple feed-forward architecture with configurable
    layers and activation functions for predicting game outcomes.
    """
    
    def __init__(
        self,
        config: ModelConfig = None,
        input_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        output_dim: Optional[int] = None
    ):
        """
        Initialize the model.
        
        Args:
            config: Model configuration
            input_dim: Number of input features (used if not in config)
            hidden_dims: List of hidden layer dimensions (used if not in config)
            dropout_rate: Dropout rate for regularization
            activation: Activation function to use ('relu', 'leaky_relu', or 'elu')
            output_dim: Number of output dimensions
        """
        # Initialize base class
        super().__init__(config=config)
        
        # Update the model type
        self.config.model_type = "basic_game_prediction"
        self.version.model_type = "basic_game_prediction"
        
        # Get hyperparameters from config or use provided values
        if config and 'hyperparameters' in vars(config):
            hyperparams = config.hyperparameters
            self.input_dim = hyperparams.get('input_size', input_dim)
            self.hidden_dims = hyperparams.get('hidden_dims', hidden_dims)
            if self.hidden_dims is None and 'hidden_size' in hyperparams:
                self.hidden_dims = [hyperparams['hidden_size']]
            self.dropout_rate = hyperparams.get('dropout_rate', dropout_rate)
            self.activation_name = hyperparams.get('activation', activation)
            self.output_dim = hyperparams.get('output_size', output_dim)
        else:
            self.input_dim = input_dim or 10
            self.hidden_dims = hidden_dims or [64, 32]
            self.dropout_rate = dropout_rate
            self.activation_name = activation
            self.output_dim = output_dim or 1
            
            # Update config hyperparameters
            if self.config:
                self.config.hyperparameters = {
                    'input_size': self.input_dim,
                    'hidden_dims': self.hidden_dims,
                    'dropout_rate': self.dropout_rate,
                    'activation': self.activation_name,
                    'output_size': self.output_dim
                }
        
        # Set activation function
        if self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()  # Default to ReLU
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the neural network architecture."""
        layers = []
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        if self.hidden_dims:
            for dim in self.hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(self.activation)
                layers.append(nn.Dropout(self.dropout_rate))
                prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        # Create the sequential model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Model output of shape [batch_size, output_dim]
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Predictions of shape [batch_size, output_dim]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return the model's hyperparameters."""
        return self.config.hyperparameters
    
    def save(self, filepath: str) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model to
            
        Returns:
            Path to the saved model file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # If filepath is a directory, create a filename
        if os.path.isdir(filepath):
            model_filename = f"basic_model_{self.version.version_id}.pt"
            filepath = os.path.join(filepath, model_filename)
        
        # Create checkpoint with model state and metadata
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'hyperparameters': self.get_hyperparameters(),
            'version': self.version.to_dict()
        }
        
        torch.save(checkpoint, filepath)
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'BasicGamePredictionModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Extract configuration
        config_dict = checkpoint.get('config', {})
        if not config_dict:
            # Legacy format compatibility
            config_dict = {
                'model_type': 'basic_game_prediction',
                'hyperparameters': checkpoint.get('hyperparameters', {})
            }
        
        # Create model config
        config = ModelConfig.from_dict(config_dict)
        
        # Create model instance
        model = cls(config=config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load version if available
        if 'version' in checkpoint:
            model.version = ModelVersion.from_dict(checkpoint['version'])
        
        return model 