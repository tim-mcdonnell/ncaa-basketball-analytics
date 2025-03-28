import pytest
import os
import torch
import tempfile
from unittest.mock import patch, MagicMock

from src.models.mlflow.tracking import (
    MLflowTracker,
    get_mlflow_client,
    setup_mlflow_tracking
)
from src.models.base import ModelConfig


class TestMLflowSetup:
    """Test suite for MLflow setup functions."""
    
    def test_setup_mlflow_tracking(self):
        """Test setting up MLflow tracking URI."""
        # Use a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock mlflow module
            with patch('src.models.mlflow.tracking.mlflow') as mock_mlflow:
                # Call the setup function
                setup_mlflow_tracking(tracking_uri=f"file://{tmp_dir}")
                
                # Verify MLflow was configured correctly
                mock_mlflow.set_tracking_uri.assert_called_once_with(f"file://{tmp_dir}")
    
    def test_get_mlflow_client(self):
        """Test creating an MLflow client."""
        # Mock MLflow modules
        with patch('src.models.mlflow.tracking.mlflow') as mock_mlflow:
            # Mock MlflowClient
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            
            # Get client
            client = get_mlflow_client()
            
            # Verify client is created
            assert client is mock_client, "Client should be the mock client"
            mock_mlflow.tracking.MlflowClient.assert_called_once()


class TestMLflowTracker:
    """Test suite for MLflowTracker class."""
    
    @pytest.fixture
    def model_config(self):
        """Create a model configuration for testing."""
        return ModelConfig(
            model_type="test_model",
            hyperparameters={
                "learning_rate": 0.01,
                "hidden_size": 32,
                "batch_size": 64
            },
            features=["feature_1", "feature_2", "feature_3"],
            training_params={
                "num_epochs": 10,
                "early_stopping_patience": 3
            }
        )
    
    @pytest.fixture
    def mock_mlflow(self):
        """Create a mock for MLflow module."""
        with patch('src.models.mlflow.tracking.mlflow') as mock_mlflow:
            # Configure mock
            mock_run = MagicMock()
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
            mock_mlflow.log_param = MagicMock()
            mock_mlflow.log_metric = MagicMock()
            mock_mlflow.log_artifact = MagicMock()
            mock_mlflow.set_tag = MagicMock()
            mock_mlflow.end_run = MagicMock()
            
            yield mock_mlflow
    
    def test_tracker_initialization(self, model_config, mock_mlflow):
        """Test initializing the MLflow tracker."""
        # Create tracker
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            model_config=model_config,
            run_name="test_run"
        )
        
        # Verify attributes
        assert tracker.experiment_name == "test_experiment", "Experiment name incorrect"
        assert tracker.model_config is model_config, "Model config incorrect"
        assert tracker.run_name == "test_run", "Run name incorrect"
    
    def test_start_run(self, model_config, mock_mlflow):
        """Test starting an MLflow run."""
        # Create tracker
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            model_config=model_config,
            run_name="test_run"
        )
        
        # Start run
        tracker.start_run()
        
        # Verify MLflow interactions
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
        
        # Verify hyperparameters are logged
        for key, value in model_config.hyperparameters.items():
            mock_mlflow.log_param.assert_any_call(key, value)
        
        # Verify tags
        mock_mlflow.set_tag.assert_any_call("model_type", model_config.model_type)
    
    def test_log_metrics(self, model_config, mock_mlflow):
        """Test logging metrics."""
        # Create tracker
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            model_config=model_config,
            run_name="test_run"
        )
        
        # Create metrics
        metrics = {
            "train_loss": 0.25,
            "val_loss": 0.30,
            "accuracy": 0.85
        }
        
        # Log metrics
        tracker.log_metrics(metrics, step=5)
        
        # Verify metrics are logged
        for key, value in metrics.items():
            mock_mlflow.log_metric.assert_any_call(key, value, step=5)
    
    def test_log_model(self, model_config, mock_mlflow):
        """Test logging a model artifact."""
        # Create mock model
        model = MagicMock()
        model.get_version.return_value.version_id = "test_model_v1"
        
        # Create tracker
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            model_config=model_config,
            run_name="test_run"
        )
        
        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pt")
            
            # Log model
            tracker.log_model(model, model_path)
            
            # Verify model is logged
            mock_mlflow.log_artifact.assert_called_once_with(model_path)
            mock_mlflow.set_tag.assert_any_call("model_version", "test_model_v1")
    
    def test_end_run(self, model_config, mock_mlflow):
        """Test ending an MLflow run."""
        # Create tracker
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            model_config=model_config,
            run_name="test_run"
        )
        
        # End run
        tracker.end_run()
        
        # Verify end_run was called
        mock_mlflow.end_run.assert_called_once()
    
    def test_context_manager(self, model_config, mock_mlflow):
        """Test using tracker as a context manager."""
        # Create tracker
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            model_config=model_config,
            run_name="test_run"
        )
        
        # Use as context manager
        with tracker:
            # Log some metrics
            tracker.log_metrics({"test_metric": 0.5}, step=1)
        
        # Verify start and end run were called
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()
        mock_mlflow.log_metric.assert_called_once_with("test_metric", 0.5, step=1) 