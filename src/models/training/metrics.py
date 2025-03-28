import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Class to track and compute training metrics."""
    
    # Track losses by epoch
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    
    # Track additional metrics (e.g., accuracy, MSE)
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    # Best validation loss and corresponding epoch
    best_val_loss: float = float('inf')
    best_epoch: int = -1
    
    def update_train_loss(self, loss: float) -> None:
        """Update training loss."""
        self.train_losses.append(loss)
    
    def update_val_loss(self, loss: float) -> None:
        """Update validation loss and check if it's the best."""
        self.val_losses.append(loss)
        
        # Check if this is the best validation loss
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_epoch = len(self.val_losses) - 1
    
    def update_train_metric(self, metric_name: str, value: float) -> None:
        """Update a training metric."""
        if metric_name not in self.train_metrics:
            self.train_metrics[metric_name] = []
        
        self.train_metrics[metric_name].append(value)
    
    def update_val_metric(self, metric_name: str, value: float) -> None:
        """Update a validation metric."""
        if metric_name not in self.val_metrics:
            self.val_metrics[metric_name] = []
        
        self.val_metrics[metric_name].append(value)
    
    def get_train_metrics(self) -> Dict[str, List[float]]:
        """
        Get all training metrics.
        
        Returns:
            Dictionary containing training loss and all tracked training metrics
        """
        metrics = {'loss': self.train_losses}
        metrics.update(self.train_metrics)
        return metrics
    
    def get_val_metrics(self) -> Dict[str, List[float]]:
        """
        Get all validation metrics.
        
        Returns:
            Dictionary containing validation loss and all tracked validation metrics
        """
        metrics = {'loss': self.val_losses}
        metrics.update(self.val_metrics)
        return metrics
    
    def set_train_metrics(self, metrics: Dict[str, List[float]]) -> None:
        """
        Set training metrics from a dictionary.
        
        Args:
            metrics: Dictionary containing training metrics
        """
        if 'loss' in metrics:
            self.train_losses = metrics['loss']
            
        # Set other metrics
        for metric_name, values in metrics.items():
            if metric_name != 'loss':
                self.train_metrics[metric_name] = values
    
    def set_val_metrics(self, metrics: Dict[str, List[float]]) -> None:
        """
        Set validation metrics from a dictionary.
        
        Args:
            metrics: Dictionary containing validation metrics
        """
        if 'loss' in metrics:
            self.val_losses = metrics['loss']
            
        # Set other metrics and recalculate best validation loss
        for metric_name, values in metrics.items():
            if metric_name != 'loss':
                self.val_metrics[metric_name] = values
        
        # Update best validation loss and epoch
        if self.val_losses:
            min_loss_idx = np.argmin(self.val_losses)
            self.best_val_loss = self.val_losses[min_loss_idx]
            self.best_epoch = min_loss_idx
    
    def get_last_epoch_metrics(self) -> Dict[str, float]:
        """Get metrics for the last epoch."""
        metrics = {}
        
        if self.train_losses:
            metrics['train_loss'] = self.train_losses[-1]
        
        if self.val_losses:
            metrics['val_loss'] = self.val_losses[-1]
        
        for metric_name, values in self.train_metrics.items():
            if values:
                metrics[f'train_{metric_name}'] = values[-1]
        
        for metric_name, values in self.val_metrics.items():
            if values:
                metrics[f'val_{metric_name}'] = values[-1]
        
        return metrics
    
    def get_best_epoch_metrics(self) -> Dict[str, float]:
        """Get metrics for the best epoch."""
        if self.best_epoch < 0:
            return {}
        
        metrics = {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        if self.train_losses and self.best_epoch < len(self.train_losses):
            metrics['best_train_loss'] = self.train_losses[self.best_epoch]
        
        for metric_name, values in self.train_metrics.items():
            if values and self.best_epoch < len(values):
                metrics[f'best_train_{metric_name}'] = values[self.best_epoch]
        
        for metric_name, values in self.val_metrics.items():
            if values and self.best_epoch < len(values):
                metrics[f'best_val_{metric_name}'] = values[self.best_epoch]
        
        return metrics
    
    def has_improved(self, patience: int) -> bool:
        """
        Check if validation loss has improved in the last `patience` epochs.
        
        Args:
            patience: Number of epochs to check for improvement
            
        Returns:
            True if the best epoch is within the last `patience` epochs
        """
        if self.best_epoch < 0:
            return True
        
        current_epoch = len(self.val_losses) - 1
        return current_epoch - self.best_epoch < patience
    
    def get_learning_curve_data(self) -> Dict[str, List[float]]:
        """Get data for plotting learning curves."""
        data = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        
        # Add additional metrics
        for metric_name, values in self.train_metrics.items():
            data[f'train_{metric_name}'] = values
        
        for metric_name, values in self.val_metrics.items():
            data[f'val_{metric_name}'] = values
        
        return data


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute binary prediction accuracy.
    
    Args:
        predictions: Predicted values (probabilities or logits)
        targets: Target values (0 or 1)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Convert predictions to binary values
    binary_preds = (predictions > 0.5).float()
    
    # Compute accuracy
    correct = (binary_preds == targets).sum().item()
    total = targets.size(0)
    
    return correct / total if total > 0 else 0.0


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        MSE as a float
    """
    return ((predictions - targets) ** 2).mean().item()


# Alias for backward compatibility with tests
calculate_mse = compute_mse


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        RMSE as a float
    """
    return torch.sqrt(((predictions - targets) ** 2).mean()).item()


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        MAE as a float
    """
    return torch.abs(predictions - targets).mean().item()


def compute_point_spread_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> float:
    """
    Compute accuracy for point spread predictions.
    
    This measures if the sign of the prediction matches the sign of the target,
    which determines if the model correctly predicted the winning team.
    
    Args:
        predictions: Predicted point spreads
        targets: Actual point spreads
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    correct = ((predictions > 0) == (targets > 0)).sum().item()
    total = targets.size(0)
    
    return correct / total if total > 0 else 0.0


def compute_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    metric_names: List[str] = ['accuracy', 'mse', 'rmse', 'mae', 'point_spread_accuracy']
) -> Dict[str, float]:
    """
    Compute multiple metrics at once.
    
    Args:
        predictions: Predicted values
        targets: Target values
        metric_names: List of metric names to compute
    
    Returns:
        Dictionary mapping metric names to values
    """
    metrics = {}
    
    if 'accuracy' in metric_names:
        metrics['accuracy'] = compute_accuracy(predictions, targets)
    
    if 'mse' in metric_names:
        metrics['mse'] = compute_mse(predictions, targets)
    
    if 'rmse' in metric_names:
        metrics['rmse'] = compute_rmse(predictions, targets)
    
    if 'mae' in metric_names:
        metrics['mae'] = compute_mae(predictions, targets)
    
    if 'point_spread_accuracy' in metric_names:
        metrics['point_spread_accuracy'] = compute_point_spread_accuracy(predictions, targets)
    
    return metrics 