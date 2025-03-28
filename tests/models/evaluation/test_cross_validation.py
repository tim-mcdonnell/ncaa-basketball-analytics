import pytest
import torch
import numpy as np
import polars as pl
from unittest.mock import Mock, patch

from src.models.evaluation.cross_validation import (
    TimeSeriesSplit,
    KFoldCrossValidator,
    CrossValidationResults,
)
from src.models.base import BaseModel, ModelConfig


class TestTimeSeriesSplit:
    """Test suite for time series cross-validation splits."""

    def test_time_series_split(self):
        """Test creating time series splits."""
        # Create mock dataset with timestamps
        data = pl.DataFrame(
            {
                "id": list(range(100)),
                "timestamp": sorted([f"2023-01-{day:02d}" for day in range(1, 101)]),
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.randn(100),
            }
        )

        # Create splitter with 5 folds and explicit test_size
        splitter = TimeSeriesSplit(n_splits=5, date_column="timestamp", test_size=20)

        # Generate splits
        splits = list(splitter.split(data))

        # Verify number of splits
        assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"

        # Verify each split is (train_indices, test_indices)
        for i, (train_idx, test_idx) in enumerate(splits):
            # Verify indices are arrays
            assert isinstance(train_idx, np.ndarray), "Train indices should be numpy array"
            assert isinstance(test_idx, np.ndarray), "Test indices should be numpy array"

            # Get train and test data
            train_data = data.filter(pl.col("id").is_in(train_idx.tolist()))
            test_data = data.filter(pl.col("id").is_in(test_idx.tolist()))

            # Verify no overlap between train and test
            assert set(train_data["id"].to_list()).isdisjoint(
                set(test_data["id"].to_list())
            ), "Train and test sets should not overlap"

            # Verify chronological order is maintained (max train date < min test date)
            max_train_date = max(train_data["timestamp"].to_list())
            min_test_date = min(test_data["timestamp"].to_list())
            assert (
                max_train_date < min_test_date
            ), f"Train dates should precede test dates, but got max train={max_train_date}, min test={min_test_date}"

            # Check sizes for reasonable splits (rough check)
            if i < 4:  # All but the last fold
                # Each fold should use approximately 1/5 of the data as test
                expected_test_size = 20
                assert (
                    abs(len(test_idx) - expected_test_size) <= 1
                ), f"Fold {i} test size {len(test_idx)} differs significantly from expected {expected_test_size}"


class TestKFoldCrossValidator:
    """Test suite for K-Fold cross-validation."""

    @pytest.fixture
    def mock_model_factory(self):
        """Create a mock model factory function."""

        def create_model(config):
            model = Mock(spec=BaseModel)
            model.config = config

            # Add parameters method
            parameters_mock = Mock()
            parameters_mock.__iter__ = Mock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))
            model.parameters = Mock(return_value=parameters_mock)

            # Make predict return reasonable values
            def predict_fn(x):
                if isinstance(x, torch.Tensor):
                    return torch.rand(x.shape[0], 1)
                return torch.rand(len(x), 1)

            model.predict = predict_fn
            return model

        return create_model

    @pytest.fixture
    def sample_data(self):
        """Create sample data for cross-validation."""
        # Create dataset with 100 samples
        data = pl.DataFrame(
            {
                "id": list(range(100)),
                "timestamp": sorted([f"2023-01-{day:02d}" for day in range(1, 101)]),
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.rand(100),  # Binary target between 0 and 1
            }
        )

        return data

    def test_kfold_validation(self, mock_model_factory, sample_data):
        """Test K-Fold cross-validation."""
        # Create model configuration
        config = ModelConfig(
            model_type="test_model",
            hyperparameters={
                "input_size": 2,
                "hidden_size": 10,
                "output_size": 1,
                "learning_rate": 0.01,
            },
            features=["feature1", "feature2"],
            training_params={"num_epochs": 5},
        )

        # Create cross-validator
        cv = KFoldCrossValidator(
            model_factory=mock_model_factory,
            model_config=config,
            data=sample_data,
            target_column="target",
            n_splits=5,
            cv_type="time_series",
            date_column="timestamp",
        )

        # Create a mocked train method return value
        mock_train_return = ([], [])  # train_losses, val_losses

        # Mock predict method to return random tensors
        def mock_predict(loader):
            return torch.rand(20, 1), torch.rand(20, 1)

        # Set up the trainer mock
        with patch("src.models.training.trainer.ModelTrainer") as MockTrainer:
            # Configure mock trainer instance
            mock_trainer_instance = Mock()
            mock_trainer_instance.train.return_value = mock_train_return
            mock_trainer_instance.predict.side_effect = mock_predict
            MockTrainer.return_value = mock_trainer_instance

            # Run cross-validation
            results = cv.run_cv()

            # Verify results object type
            assert isinstance(
                results, CrossValidationResults
            ), "Should return CrossValidationResults"

            # Verify fold metrics
            assert (
                len(results.fold_metrics) == 5
            ), f"Expected metrics for 5 folds, got {len(results.fold_metrics)}"

            # Verify each fold has metrics
            for metrics in results.fold_metrics:
                assert "accuracy" in metrics, "Fold metrics should include accuracy"
                assert "mse" in metrics, "Fold metrics should include MSE"
                assert "rmse" in metrics, "Fold metrics should include RMSE"
                assert "mae" in metrics, "Fold metrics should include MAE"

            # Verify aggregate metrics
            assert (
                "mean_accuracy" in results.aggregate_metrics
            ), "Aggregate metrics should include mean accuracy"
            assert (
                "std_accuracy" in results.aggregate_metrics
            ), "Aggregate metrics should include std accuracy"


class TestCrossValidationResults:
    """Test suite for cross-validation results handling."""

    def test_results_aggregation(self):
        """Test aggregating results across folds."""
        # Create fold metrics
        fold_metrics = [
            {"accuracy": 0.80, "rmse": 0.5, "calibration_score": 0.90},
            {"accuracy": 0.75, "rmse": 0.6, "calibration_score": 0.85},
            {"accuracy": 0.82, "rmse": 0.45, "calibration_score": 0.92},
            {"accuracy": 0.78, "rmse": 0.55, "calibration_score": 0.88},
            {"accuracy": 0.77, "rmse": 0.52, "calibration_score": 0.87},
        ]

        # Create results object
        results = CrossValidationResults(fold_metrics=fold_metrics)

        # Verify aggregate metrics are calculated
        agg_metrics = results.aggregate_metrics

        # Check values
        assert "mean_accuracy" in agg_metrics, "Missing mean accuracy"
        assert "std_accuracy" in agg_metrics, "Missing accuracy standard deviation"
        assert "mean_rmse" in agg_metrics, "Missing mean RMSE"

        # Calculate expected values
        expected_mean_accuracy = sum(fold["accuracy"] for fold in fold_metrics) / len(fold_metrics)
        expected_mean_rmse = sum(fold["rmse"] for fold in fold_metrics) / len(fold_metrics)

        # Verify accuracy
        assert np.isclose(
            agg_metrics["mean_accuracy"], expected_mean_accuracy
        ), f"Expected mean accuracy {expected_mean_accuracy}, got {agg_metrics['mean_accuracy']}"
        assert np.isclose(
            agg_metrics["mean_rmse"], expected_mean_rmse
        ), f"Expected mean RMSE {expected_mean_rmse}, got {agg_metrics['mean_rmse']}"

    def test_results_summary(self):
        """Test generating results summary."""
        # Create fold metrics
        fold_metrics = [
            {"accuracy": 0.80, "rmse": 0.5},
            {"accuracy": 0.75, "rmse": 0.6},
            {"accuracy": 0.82, "rmse": 0.45},
        ]

        # Create results with feature importance
        feature_importance = {"feature1": 0.4, "feature2": 0.3, "feature3": 0.2, "feature4": 0.1}

        # Create results object
        results = CrossValidationResults(
            fold_metrics=fold_metrics, feature_importance=feature_importance
        )

        # Get summary
        summary = results.get_summary()

        # Verify summary structure
        assert "aggregate_metrics" in summary, "Missing aggregate metrics in summary"
        assert "feature_importance" in summary, "Missing feature importance in summary"
        assert "fold_metrics" in summary, "Missing fold metrics in summary"

        # Verify feature importance order (should be sorted by importance)
        sorted_features = summary["feature_importance"]
        assert sorted_features[0][0] == "feature1", "First feature should be feature1"
        assert sorted_features[1][0] == "feature2", "Second feature should be feature2"
