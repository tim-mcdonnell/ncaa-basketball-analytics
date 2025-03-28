import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import os
import copy
from .metrics import TrainingMetrics, compute_metrics
from src.models.base import ModelConfig


class ModelTrainer:
    """
    Class to handle model training, validation, and evaluation.
    
    This class provides a unified interface for training models, tracking
    metrics, and implementing early stopping.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig
    ):
        """
        Initialize the model trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader containing training data
            val_loader: DataLoader containing validation data
            config: ModelConfig object containing training parameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Get training parameters from config
        training_params = config.training_params
        learning_rate = config.hyperparameters.get("learning_rate", 0.001)
        weight_decay = config.hyperparameters.get("weight_decay", 0.0)
        
        # Set device
        self.device = torch.device(training_params.get("device", "cpu") 
                                   if training_params.get("device") 
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Configure loss function (default to MSE for regression tasks)
        loss_fn_name = training_params.get("loss_function", "mse")
        self.criterion = self._get_loss_function(loss_fn_name)
        
        # Configure optimizer
        optimizer_name = training_params.get("optimizer", "adam")
        self.optimizer = self._get_optimizer(
            optimizer_name, 
            learning_rate, 
            weight_decay
        )
        
        # Set metrics to track
        self.metrics_to_track = training_params.get("metrics_to_track", 
                                               ['accuracy', 'mse', 'point_spread_accuracy'])
        
        # Initialize metrics tracker
        self.metrics = TrainingMetrics()
        
        # Set checkpoint directory
        self.checkpoint_dir = training_params.get("model_dir", './checkpoints')
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _get_loss_function(self, loss_name: str) -> nn.Module:
        """Get loss function by name."""
        loss_map = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'bce': nn.BCELoss(),
            'ce': nn.CrossEntropyLoss()
        }
        return loss_map.get(loss_name.lower(), nn.MSELoss())
    
    def _get_optimizer(self, 
                      optimizer_name: str, 
                      learning_rate: float, 
                      weight_decay: float) -> optim.Optimizer:
        """Get optimizer by name."""
        optimizer_map = {
            'adam': optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            ),
            'sgd': optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            ),
            'rmsprop': optim.RMSprop(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        }
        return optimizer_map.get(optimizer_name.lower(), 
                               optim.Adam(self.model.parameters(), lr=learning_rate))
    
    def _get_device(self) -> torch.device:
        """Get the best available device (CUDA if available, otherwise CPU)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
            
        Returns:
            Average loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch in self.train_loader:
            # Get data and move to device
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track loss and predictions for metrics
            total_loss += loss.item() * inputs.size(0)
            all_predictions.append(outputs.detach())
            all_targets.append(targets)
        
        # Compute average loss
        avg_loss = total_loss / len(self.train_loader.dataset)
        self.metrics.update_train_loss(avg_loss)
        
        # Compute other metrics if there's data
        if all_predictions and all_targets:
            predictions = torch.cat(all_predictions)
            targets = torch.cat(all_targets)
            
            epoch_metrics = compute_metrics(
                predictions, 
                targets, 
                self.metrics_to_track
            )
            
            # Update metrics
            for metric_name, value in epoch_metrics.items():
                self.metrics.update_train_metric(metric_name, value)
        
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model on the validation set.
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get data and move to device
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track loss and predictions for metrics
                total_loss += loss.item() * inputs.size(0)
                all_predictions.append(outputs)
                all_targets.append(targets)
        
        # Compute average loss
        avg_loss = total_loss / len(self.val_loader.dataset)
        self.metrics.update_val_loss(avg_loss)
        
        # Compute other metrics if there's data
        if all_predictions and all_targets:
            predictions = torch.cat(all_predictions)
            targets = torch.cat(all_targets)
            
            epoch_metrics = compute_metrics(
                predictions, 
                targets, 
                self.metrics_to_track
            )
            
            # Update metrics
            for metric_name, value in epoch_metrics.items():
                self.metrics.update_val_metric(metric_name, value)
        
        return avg_loss
    
    def train(self) -> Tuple[List[float], List[float]]:
        """
        Train the model for multiple epochs.
            
        Returns:
            Tuple of (train_losses, val_losses) lists
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        # Get training parameters from config
        num_epochs = self.config.training_params.get("num_epochs", 10)
        patience = self.config.training_params.get("early_stopping_patience", 5)
        
        for epoch in range(num_epochs):
            # Train one epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_weights = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(epoch)
            else:
                patience_counter += 1
                
            # Check for early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            
        return train_losses, val_losses
    
    def _save_checkpoint(self, epoch: int) -> str:
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': self.metrics.get_train_metrics(),
            'val_metrics': self.metrics.get_val_metrics(),
        }, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Epoch number of the loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore metrics if available
        if 'train_metrics' in checkpoint:
            self.metrics.set_train_metrics(checkpoint['train_metrics'])
        if 'val_metrics' in checkpoint:
            self.metrics.set_val_metrics(checkpoint['val_metrics'])
            
        return checkpoint['epoch']
    
    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the model.
        
        Args:
            data_loader: DataLoader containing data to predict on
            
        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                all_predictions.append(outputs.cpu())
                all_targets.append(targets)
        
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        
        return predictions, targets
    
    def get_metrics(self) -> TrainingMetrics:
        """Get the training metrics object."""
        return self.metrics 