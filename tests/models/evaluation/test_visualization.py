import pytest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.models.evaluation.visualization import (
    plot_learning_curves,
    plot_feature_importance,
    plot_calibration_curve,
    plot_confusion_matrix,
    save_evaluation_plots
)


class TestVisualizationFunctions:
    """Test suite for model evaluation visualization functions."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample training metrics data."""
        # Training data over 10 epochs
        train_losses = [2.5, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        val_losses = [2.6, 2.3, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.25, 1.2]
        
        train_metrics = {
            "accuracy": [0.60, 0.65, 0.70, 0.73, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86],
            "rmse": [0.9, 0.85, 0.8, 0.75, 0.7, 0.68, 0.65, 0.63, 0.61, 0.60]
        }
        
        val_metrics = {
            "accuracy": [0.58, 0.63, 0.68, 0.71, 0.74, 0.76, 0.77, 0.78, 0.79, 0.80],
            "rmse": [0.92, 0.87, 0.82, 0.78, 0.74, 0.72, 0.70, 0.68, 0.67, 0.65]
        }
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "epochs": list(range(1, 11))
        }
    
    @pytest.fixture
    def sample_feature_importance(self):
        """Create sample feature importance data."""
        return {
            "team_win_ratio": 0.25,
            "point_differential": 0.20,
            "offensive_rating": 0.15,
            "defensive_rating": 0.15,
            "turnover_percentage": 0.10,
            "rebound_percentage": 0.10,
            "free_throw_rate": 0.05
        }
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data for calibration curve."""
        # 100 samples
        predictions = torch.tensor(np.linspace(0.1, 0.9, 100))
        
        # Actuals - higher prediction means higher chance of being actual 1
        actuals = torch.zeros_like(predictions)
        actuals[predictions > 0.7] = 1.0
        
        # Add some randomness in middle range
        middle_mask = (predictions > 0.3) & (predictions < 0.7)
        middle_count = middle_mask.sum().item()
        actuals[middle_mask] = torch.tensor(
            np.random.choice([0.0, 1.0], size=middle_count, 
                             p=[0.7, 0.3])  # Slight bias toward 0
        )
        
        return predictions, actuals
    
    @pytest.fixture
    def sample_confusion(self):
        """Create sample confusion matrix."""
        # Create a 2x2 confusion matrix
        # Format: [[TP, FP], [FN, TN]]
        return np.array([
            [85, 10],
            [15, 90]
        ])
    
    def test_plot_learning_curves(self, sample_metrics):
        """Test plotting learning curves."""
        # Mock the plotting functions
        with patch('src.models.evaluation.visualization.plt') as mock_plt, \
             patch('src.models.evaluation.visualization.Figure') as mock_fig:
            
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_plt.figure.return_value = mock_fig_instance
            mock_ax = MagicMock()
            mock_fig_instance.add_subplot.return_value = mock_ax
            
            # Call the function
            fig = plot_learning_curves(
                train_losses=sample_metrics["train_losses"],
                val_losses=sample_metrics["val_losses"],
                train_metrics=sample_metrics["train_metrics"],
                val_metrics=sample_metrics["val_metrics"],
                epochs=sample_metrics["epochs"]
            )
            
            # Verify figure creation
            mock_plt.figure.assert_called_once()
            
            # Verify plot calls - should create at least 3 subplots (loss + 2 metrics)
            assert mock_fig_instance.add_subplot.call_count >= 3, \
                "Should create at least 3 subplots"
            
            # Verify axes were used for plotting
            assert mock_ax.plot.call_count >= 6, \
                "Should plot at least 6 lines (train/val for loss and each metric)"
            
            # Verify labels and legends
            assert mock_ax.set_xlabel.call_count >= 3, "Should set x labels"
            assert mock_ax.set_ylabel.call_count >= 3, "Should set y labels"
            assert mock_ax.legend.call_count >= 3, "Should create legends"
    
    def test_plot_feature_importance(self, sample_feature_importance):
        """Test plotting feature importance."""
        # Mock the plotting functions
        with patch('src.models.evaluation.visualization.plt') as mock_plt, \
             patch('src.models.evaluation.visualization.Figure') as mock_fig:
            
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_plt.figure.return_value = mock_fig_instance
            mock_ax = MagicMock()
            mock_fig_instance.add_subplot.return_value = mock_ax
            
            # Call the function
            fig = plot_feature_importance(sample_feature_importance)
            
            # Verify figure creation
            mock_plt.figure.assert_called_once()
            
            # Verify barplot creation
            mock_ax.barh.assert_called_once()
            
            # Verify the number of features plotted
            call_args = mock_ax.barh.call_args[0]
            assert len(call_args[0]) == len(sample_feature_importance), \
                "Should plot all features"
            
            # Verify sorting - most important feature should be at the top (reversed order)
            # The y-position should match the importance order
            feature_names = call_args[0]
            assert feature_names[0] == "team_win_ratio", \
                "Most important feature should be at the top"
    
    def test_plot_calibration_curve(self, sample_predictions):
        """Test plotting calibration curve."""
        predictions, actuals = sample_predictions
        
        # Mock the plotting functions
        with patch('src.models.evaluation.visualization.plt') as mock_plt, \
             patch('src.models.evaluation.visualization.Figure') as mock_fig:
            
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_plt.figure.return_value = mock_fig_instance
            mock_ax = MagicMock()
            mock_fig_instance.add_subplot.return_value = mock_ax
            
            # Call the function
            fig = plot_calibration_curve(predictions, actuals)
            
            # Verify figure creation
            mock_plt.figure.assert_called_once()
            
            # Verify plot and reference line
            assert mock_ax.plot.call_count >= 2, \
                "Should plot at least 2 lines (calibration curve and reference)"
            
            # Verify labels and title
            mock_ax.set_xlabel.assert_called_once()
            mock_ax.set_ylabel.assert_called_once()
            mock_ax.set_title.assert_called_once()
    
    def test_plot_confusion_matrix(self, sample_confusion):
        """Test plotting confusion matrix."""
        # Mock the plotting functions
        with patch('src.models.evaluation.visualization.plt') as mock_plt, \
             patch('src.models.evaluation.visualization.Figure') as mock_fig, \
             patch('src.models.evaluation.visualization.sns') as mock_sns:
            
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_plt.figure.return_value = mock_fig_instance
            mock_ax = MagicMock()
            mock_fig_instance.add_subplot.return_value = mock_ax
            
            # Call the function
            fig = plot_confusion_matrix(
                sample_confusion,
                class_names=["Win", "Loss"]
            )
            
            # Verify figure creation
            mock_plt.figure.assert_called_once()
            
            # Verify heatmap creation
            mock_sns.heatmap.assert_called_once()
            
            # Verify correct confusion matrix was used
            call_args = mock_sns.heatmap.call_args[0]
            np.testing.assert_array_equal(call_args[0], sample_confusion)
            
            # Verify labels and title
            mock_ax.set_xlabel.assert_called_once()
            mock_ax.set_ylabel.assert_called_once()
            mock_ax.set_title.assert_called_once()
    
    def test_save_evaluation_plots(self, sample_metrics, sample_feature_importance, sample_confusion, sample_predictions):
        """Test saving all evaluation plots."""
        # Create mock figures
        mock_figures = {
            "learning_curves": MagicMock(),
            "feature_importance": MagicMock(),
            "calibration_curve": MagicMock(),
            "confusion_matrix": MagicMock()
        }
        
        # Mock the plotting functions to return our mock figures
        with patch('src.models.evaluation.visualization.plot_learning_curves', 
                   return_value=mock_figures["learning_curves"]), \
             patch('src.models.evaluation.visualization.plot_feature_importance',
                   return_value=mock_figures["feature_importance"]), \
             patch('src.models.evaluation.visualization.plot_calibration_curve',
                   return_value=mock_figures["calibration_curve"]), \
             patch('src.models.evaluation.visualization.plot_confusion_matrix',
                   return_value=mock_figures["confusion_matrix"]):
            
            # Create a temporary directory for saving
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Call the function to save plots
                save_paths = save_evaluation_plots(
                    output_dir=tmp_dir,
                    train_losses=sample_metrics["train_losses"],
                    val_losses=sample_metrics["val_losses"],
                    train_metrics=sample_metrics["train_metrics"],
                    val_metrics=sample_metrics["val_metrics"],
                    epochs=sample_metrics["epochs"],
                    feature_importance=sample_feature_importance,
                    predictions=sample_predictions[0],
                    actuals=sample_predictions[1],
                    confusion_matrix=sample_confusion,
                    class_names=["Win", "Loss"]
                )
                
                # Verify all figures were saved
                for plot_type, fig in mock_figures.items():
                    fig.savefig.assert_called_once()
                
                # Verify return value contains all paths
                assert "learning_curves" in save_paths, "Missing learning curves path"
                assert "feature_importance" in save_paths, "Missing feature importance path"
                assert "calibration_curve" in save_paths, "Missing calibration curve path"
                assert "confusion_matrix" in save_paths, "Missing confusion matrix path" 