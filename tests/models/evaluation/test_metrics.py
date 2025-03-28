import pytest
import torch
import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix

from src.models.evaluation.metrics import (
    calculate_prediction_accuracy,
    calculate_point_spread_accuracy,
    calculate_calibration_metrics,
    calculate_feature_importance,
    EvaluationMetrics
)


class TestEvaluationMetrics:
    """Tests for evaluation metric functions."""
    
    def test_prediction_accuracy(self):
        """Test binary prediction accuracy calculation."""
        # Create test predictions and actual values
        y_pred = torch.tensor([0.8, 0.6, 0.4, 0.3, 0.9, 0.2, 0.7, 0.1])
        y_true = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        
        # Calculate accuracy
        accuracy = calculate_prediction_accuracy(y_pred, y_true)
        
        # Expected results: 6 out of 8 correct (0.75)
        # Correctly predicted 0.8->1, 0.6->1, 0.4->0, 0.3->0, 0.2->0, 0.1->1(wrong)
        expected_accuracy = 0.75
        
        # Verify results
        assert isinstance(accuracy, float), "Accuracy should be a float"
        assert np.isclose(accuracy, expected_accuracy), \
            f"Expected accuracy {expected_accuracy}, got {accuracy}"
    
    def test_point_spread_accuracy(self):
        """Test calculation of point spread accuracy."""
        # Create test data
        # Positive spread means home team is predicted/actual winner
        predicted_spreads = torch.tensor([5.0, -3.0, 7.0, -2.0, 1.0, -1.0, 3.0, -4.0])
        actual_spreads = torch.tensor([3.0, -5.0, 10.0, 1.0, -1.0, -2.0, 2.0, -6.0])
        
        # Calculate accuracy
        accuracy, detailed_results = calculate_point_spread_accuracy(
            predicted_spreads, 
            actual_spreads,
            return_details=True
        )
        
        # Expected results
        # Correct winner predictions: indices 0, 1, 2, 5, 6, 7 (6 out of 8)
        expected_accuracy = 6/8
        
        # Verify results
        assert isinstance(accuracy, float), "Accuracy should be a float"
        assert np.isclose(accuracy, expected_accuracy), \
            f"Expected accuracy {expected_accuracy}, got {accuracy}"
        assert len(detailed_results) == 8, "Should have details for each prediction"
        
        # Verify details contain correct information
        assert "correct_winner" in detailed_results[0], "Details missing correct_winner"
        assert "prediction_error" in detailed_results[0], "Details missing prediction_error"
        
        # Check first prediction details
        assert detailed_results[0]["correct_winner"] == True, "First prediction should be correct"
        assert "prediction_error" in detailed_results[0], "Missing prediction error"
    
    def test_calibration_metrics(self):
        """Test calibration metrics calculation."""
        # Create predictions in 5 distinct ranges
        predictions = torch.tensor([
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # Range 0.0-0.2
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,  # Range 0.2-0.4
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # Range 0.4-0.6
            0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,  # Range 0.6-0.8
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,  # Range 0.8-1.0
        ])
        
        # Create actuals with progressively higher positive rates
        actuals = torch.zeros_like(predictions)
        actuals[0:2] = 1.0    # 20% in first bin
        actuals[10:14] = 1.0  # 40% in second bin
        actuals[20:25] = 1.0  # 50% in third bin
        actuals[30:37] = 1.0  # 70% in fourth bin
        actuals[40:49] = 1.0  # 90% in fifth bin
        
        # Calculate calibration metrics
        metrics = calculate_calibration_metrics(actuals, predictions, n_bins=5)
        
        # Check that the metrics object has the expected structure
        assert len(metrics['bins']) == 5, "Should have 5 calibration bins"
        assert 'brier_score' in metrics, "Missing brier score"
        assert 'calibration_error' in metrics, "Missing calibration error"
        assert 'calibration_curve' in metrics, "Missing calibration curve"
        assert 'bins' in metrics, "Missing bins data"
        assert 'confusion_matrix' in metrics, "Missing confusion matrix"
        
        # Verify bin structure
        for bin_data in metrics['bins']:
            assert 'bin_start' in bin_data, "Bin missing start value"
            assert 'bin_end' in bin_data, "Bin missing end value"
            assert 'pred_mean' in bin_data, "Bin missing prediction mean"
            assert 'actual_mean' in bin_data, "Bin missing actual mean"
            assert 'samples' in bin_data, "Bin missing sample count"
            assert bin_data['bin_start'] < bin_data['bin_end'], "Bin boundaries incorrect"

    def test_feature_importance(self):
        """Test calculation of feature importance."""
        # Create mock model with fixed importances
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.1, 0.2, 0.05, 0.5, 0.15])
                
            def predict(self, x):
                # Simple model that doesn't matter for this test
                # since we're using the built-in feature_importances_
                return torch.ones(x.shape[0], 1)

        model = MockModel()

        # Create test data - values don't matter since we use built-in importances
        features = torch.tensor([
            [1.0, 2.0, 3.0, 0.5, 1.5],
            [2.0, 1.0, 0.0, 1.0, 2.0],
            [0.5, 1.5, 2.5, 0.0, 1.0],
            [1.5, 0.5, 1.0, 2.0, 0.0]
        ])

        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

        # Calculate feature importance
        importance = calculate_feature_importance(model, features, feature_names)

        # Verify structure
        assert isinstance(importance, dict), "Importance should be a dictionary"
        assert all(name in importance for name in feature_names), "All feature names should be in results"

        # Given the mock model, feature 4 should have the highest importance
        assert importance["feature_4"] > importance["feature_1"], \
            f"Feature 4 should have higher importance than feature 1: {importance['feature_4']} vs {importance['feature_1']}"


class TestEvaluationMetricsClass:
    """Test suite for the EvaluationMetrics class."""

    @pytest.fixture
    def sample_predictions(self):
        # Generate 100 samples with probabilities from 0 to 1
        torch.manual_seed(42)  # Set seed for reproducibility
        predictions = torch.linspace(0, 1, 100)
        
        # Create actual values: 
        # - predictions below 0.3 are mostly 0 (80%)
        # - predictions above 0.7 are mostly 1 (80%)
        # - predictions in between are 50/50
        actuals = torch.zeros_like(predictions)
        
        # Low range (below 0.3) - mostly 0s with some 1s
        low_mask = predictions < 0.3
        actuals[low_mask] = 0
        # Randomly set 20% to 1
        idx = torch.where(low_mask)[0]
        idx = idx[torch.randperm(len(idx))[:int(len(idx) * 0.2)]]
        actuals[idx] = 1
        
        # High range (above 0.7) - mostly 1s with some 0s
        high_mask = predictions > 0.7
        actuals[high_mask] = 1
        # Randomly set 20% to 0
        idx = torch.where(high_mask)[0]
        idx = idx[torch.randperm(len(idx))[:int(len(idx) * 0.2)]]
        actuals[idx] = 0
        
        # Middle range (0.3-0.7) - 50/50 split
        mid_mask = ~(low_mask | high_mask)
        mid_indices = torch.where(mid_mask)[0]
        actuals[mid_indices[:len(mid_indices)//2]] = 0
        actuals[mid_indices[len(mid_indices)//2:]] = 1
        
        # Convert to expected format
        y_pred = predictions.reshape(-1, 1)  # Shape: [100, 1]
        y_true = actuals.reshape(-1, 1)      # Shape: [100, 1]
        
        return {
            'y_pred': y_pred,
            'y_true': y_true,
            'feature_names': ['feature1']
        }

    def test_metrics_calculation(self, sample_predictions):
        """Test calculating all metrics."""
        predictions, actuals = sample_predictions['y_pred'], sample_predictions['y_true']
        
        # Create feature data
        n_samples = len(predictions)
        n_features = 5
        features = torch.randn(n_samples, n_features)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Create mock model
        class MockModel:
            def predict(self, x):
                # Simple linear model
                weights = torch.tensor([0.2, 0.3, 0.1, 0.25, 0.15])
                return torch.sigmoid(torch.sum(x * weights, dim=1, keepdim=True))
        
        model = MockModel()
        
        # Create metrics calculator
        metrics = EvaluationMetrics(
            predictions=predictions,
            actuals=actuals,
            features=features,
            feature_names=feature_names,
            model=model
        )
        
        # Calculate all metrics
        results = metrics.calculate_all_metrics()
        
        # Verify results structure
        assert "accuracy" in results, "Missing accuracy metric"
        assert "calibration" in results, "Missing calibration metrics"
        assert "feature_importance" in results, "Missing feature importance"
        
        # Verify accuracy is between 0 and 1
        assert 0 <= results["accuracy"] <= 1, "Accuracy should be between 0 and 1"
        
        # Verify feature importance includes all features
        assert all(name in results["feature_importance"] for name in feature_names), \
            "Feature importance should include all features"
        
        # Verify report method works
        report = metrics.get_report()
        assert isinstance(report, dict), "Report should be a dictionary"
        assert "metrics" in report, "Report should contain metrics"
        assert "metadata" in report, "Report should contain metadata" 