import torch
import numpy as np

from src.models.training.metrics import (
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_accuracy,
    compute_point_spread_accuracy,
    TrainingMetrics,
)


class TestTrainingMetrics:
    """Test suite for training metrics calculation."""

    def test_mean_squared_error(self):
        """Test MSE calculation."""
        # Create test data
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.2])

        # Calculate MSE
        mse = compute_mse(predictions, targets)

        # Calculate expected result manually
        expected_mse = (
            (1.0 - 1.1) ** 2
            + (2.0 - 1.9) ** 2
            + (3.0 - 3.2) ** 2
            + (4.0 - 3.8) ** 2
            + (5.0 - 5.2) ** 2
        ) / 5

        # Verify result
        assert isinstance(mse, float), "MSE should be a float"
        assert np.isclose(mse, expected_mse), f"Expected MSE {expected_mse}, got {mse}"

    def test_root_mean_squared_error(self):
        """Test RMSE calculation."""
        # Create test data
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.2])

        # Calculate RMSE
        rmse = compute_rmse(predictions, targets)

        # Calculate expected result manually
        expected_mse = (
            (1.0 - 1.1) ** 2
            + (2.0 - 1.9) ** 2
            + (3.0 - 3.2) ** 2
            + (4.0 - 3.8) ** 2
            + (5.0 - 5.2) ** 2
        ) / 5
        expected_rmse = np.sqrt(expected_mse)

        # Verify result
        assert isinstance(rmse, float), "RMSE should be a float"
        assert np.isclose(rmse, expected_rmse), f"Expected RMSE {expected_rmse}, got {rmse}"

    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        # Create test data
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.2])

        # Calculate MAE
        mae = compute_mae(predictions, targets)

        # Calculate expected result manually
        expected_mae = (
            abs(1.0 - 1.1) + abs(2.0 - 1.9) + abs(3.0 - 3.2) + abs(4.0 - 3.8) + abs(5.0 - 5.2)
        ) / 5

        # Verify result
        assert isinstance(mae, float), "MAE should be a float"
        assert np.isclose(mae, expected_mae), f"Expected MAE {expected_mae}, got {mae}"

    def test_prediction_accuracy(self):
        """Test binary prediction accuracy calculation."""
        # Create test data - binary classification (win/loss)
        predictions = torch.tensor([0.7, 0.3, 0.6, 0.2, 0.8])  # Probabilities
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])  # Actual results (1=win, 0=loss)

        # Calculate accuracy
        accuracy = compute_accuracy(predictions, targets)

        # Expected result: 5 out of 5 correct (1.0)
        # The compute_accuracy function uses a threshold of 0.5 for binary classification
        expected_accuracy = 1.0

        # Verify result
        assert isinstance(accuracy, float), "Accuracy should be a float"
        assert np.isclose(
            accuracy, expected_accuracy
        ), f"Expected accuracy {expected_accuracy}, got {accuracy}"

    def test_point_spread_accuracy(self):
        """Test point spread accuracy calculation."""
        # Create test data - predicted vs actual point spreads
        # Positive = home team wins by that many points
        predicted_spreads = torch.tensor([5.0, -3.0, 7.0, -2.0, 1.0])
        actual_spreads = torch.tensor([3.0, -5.0, 10.0, 2.0, -1.0])

        # Calculate point spread accuracy
        accuracy = compute_point_spread_accuracy(predicted_spreads, actual_spreads)

        # Check if predictions correctly predicted the winner (sign matches)
        # 3 out of 5 match signs (5.0~3.0, -3.0~-5.0, 7.0~10.0)
        expected_accuracy = 0.6

        # Verify result
        assert isinstance(accuracy, float), "Point spread accuracy should be a float"
        assert np.isclose(
            accuracy, expected_accuracy
        ), f"Expected accuracy {expected_accuracy}, got {accuracy}"


class TestMetricsTracker:
    """Test suite for the metrics tracker."""

    def test_metrics_tracker_initialization(self):
        """Test initializing the metrics tracker."""
        # Create tracker
        tracker = TrainingMetrics()

        # Verify initial state
        assert tracker.train_losses == [], "Train losses should start empty"
        assert tracker.val_losses == [], "Validation losses should start empty"
        assert tracker.train_metrics == {}, "Train metrics should start empty"
        assert tracker.val_metrics == {}, "Validation metrics should start empty"

    def test_update_training_metrics(self):
        """Test updating training metrics."""
        # Create tracker
        tracker = TrainingMetrics()

        # Update with first epoch data
        loss = 2.5
        tracker.update_train_loss(loss)
        tracker.update_train_metric("mse", 2.5)
        tracker.update_train_metric("rmse", 1.58)
        tracker.update_train_metric("accuracy", 0.65)

        # Verify updates
        assert len(tracker.train_losses) == 1, "Should have 1 training loss"
        assert tracker.train_losses[0] == 2.5, "Training loss incorrect"
        assert "mse" in tracker.train_metrics, "MSE missing from metrics"
        assert tracker.train_metrics["mse"] == [2.5], "MSE value incorrect"
        assert "rmse" in tracker.train_metrics, "RMSE missing from metrics"
        assert "accuracy" in tracker.train_metrics, "Accuracy missing from metrics"

        # Update with second epoch data
        tracker.update_train_loss(2.2)
        tracker.update_train_metric("mse", 2.2)
        tracker.update_train_metric("rmse", 1.48)
        tracker.update_train_metric("accuracy", 0.68)

        # Verify second update
        assert len(tracker.train_losses) == 2, "Should have 2 training losses"
        assert tracker.train_losses[1] == 2.2, "Second training loss incorrect"
        assert len(tracker.train_metrics["mse"]) == 2, "Should have 2 MSE values"
        assert tracker.train_metrics["mse"][1] == 2.2, "Second MSE value incorrect"

    def test_update_validation_metrics(self):
        """Test updating validation metrics."""
        # Create tracker
        tracker = TrainingMetrics()

        # Update with validation data
        tracker.update_val_loss(2.6)
        tracker.update_val_metric("mse", 2.6)
        tracker.update_val_metric("rmse", 1.61)
        tracker.update_val_metric("accuracy", 0.63)

        # Verify updates
        assert len(tracker.val_losses) == 1, "Should have 1 validation loss"
        assert tracker.val_losses[0] == 2.6, "Validation loss incorrect"
        assert "mse" in tracker.val_metrics, "MSE missing from validation metrics"
        assert "rmse" in tracker.val_metrics, "RMSE missing from validation metrics"

        # Update with second validation data
        tracker.update_val_loss(2.4)
        tracker.update_val_metric("mse", 2.4)
        tracker.update_val_metric("rmse", 1.55)
        tracker.update_val_metric("accuracy", 0.64)

        # Verify second update
        assert len(tracker.val_losses) == 2, "Should have 2 validation losses"
        assert tracker.val_metrics["accuracy"] == [0.63, 0.64], "Accuracy values incorrect"

    def test_get_best_epoch(self):
        """Test getting the epoch with best validation performance."""
        # Create tracker
        tracker = TrainingMetrics()

        # Add data for multiple epochs
        val_losses = [2.5, 2.3, 2.1, 2.2, 2.4]
        for loss in val_losses:
            tracker.update_val_loss(loss)

        # Verify - epoch 2 (index 2) has lowest loss (2.1)
        assert tracker.best_epoch == 2, f"Expected best epoch 2, got {tracker.best_epoch}"
        assert tracker.best_val_loss == 2.1, f"Expected best loss 2.1, got {tracker.best_val_loss}"
