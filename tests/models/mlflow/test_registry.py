import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.models.mlflow.registry import (
    register_model,
    get_latest_model_version,
    list_model_versions,
    load_registered_model
)
from src.models.base import ModelConfig, ModelVersion


class TestRegistryFunctions:
    """Test suite for the MLflow registry functions."""
    
    @pytest.fixture
    def mock_mlflow(self):
        """Create mock for MLflow module."""
        with patch('src.models.mlflow.registry.mlflow') as mock_mlflow:
            # Configure mock client
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            
            # Configure mock for mlflow.register_model
            mock_model_details = MagicMock()
            mock_model_details.version = "1"
            mock_mlflow.register_model.return_value = mock_model_details
            
            # Configure mock for pytorch logging
            mock_mlflow.pytorch = MagicMock()
            
            # Configure mock run
            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_id"
            mock_client.get_run.return_value = mock_run
            
            # Configure model version
            mock_model_version = MagicMock()
            mock_model_version.version = "1"
            mock_model_version.name = "test_model"
            mock_model_version.run_id = "test_run_id"
            mock_model_version.current_stage = "Production"
            mock_model_version.creation_timestamp = 1000
            mock_model_version.last_updated_timestamp = 1100
            mock_model_version.description = "Test model description"
            
            # Configure model versions search
            mock_client.search_model_versions.return_value = [mock_model_version]
            
            yield mock_mlflow
    
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
    
    def test_register_model_from_run(self, mock_mlflow):
        """Test registering a model from a run."""
        # Register the model from a run path
        model_uri = register_model(
            model_path="runs:/test_run_id/model",
            name="test_model",
            description="Test model description",
            tags={"key": "value"}
        )
        
        # Verify interactions with MLflow
        mock_mlflow.register_model.assert_called_once_with(
            model_uri="runs:/test_run_id/model",
            name="test_model"
        )
        
        # Verify returned URI
        assert model_uri == "models:/test_model/1", f"Expected 'models:/test_model/1', got {model_uri}"
    
    def test_get_latest_model_version(self, mock_mlflow):
        """Test retrieving the latest model version."""
        # Get the latest model version
        version_info = get_latest_model_version(name="test_model")
        
        # Verify interactions with MLflow
        client = mock_mlflow.tracking.MlflowClient.return_value
        client.search_model_versions.assert_called_once_with(
            f"name='test_model'"
        )
        
        # Verify returned version info
        assert version_info["name"] == "test_model", "Model name incorrect"
        assert version_info["version"] == "1", "Version incorrect" 
        assert version_info["stage"] == "Production", "Stage incorrect"
        assert version_info["run_id"] == "test_run_id", "Run ID incorrect"
    
    def test_list_model_versions(self, mock_mlflow):
        """Test listing all versions of a model."""
        # Configure mock for model versions
        client = mock_mlflow.tracking.MlflowClient.return_value
        
        # Create mock versions
        mock_versions = []
        for i in range(1, 4):
            mv = MagicMock()
            mv.version = str(i)
            mv.name = "test_model"
            mv.run_id = f"test_run_{i}"
            mv.current_stage = "Production"
            mv.creation_timestamp = 1000 * i  # Increasing timestamps
            mv.last_updated_timestamp = 1100 * i
            mv.description = f"Version {i}"
            mock_versions.append(mv)
        
        client.search_model_versions.return_value = mock_versions
        
        # List model versions
        versions = list_model_versions(name="test_model")
        
        # Verify interactions with MLflow
        client.search_model_versions.assert_called_once_with(
            f"name='test_model'"
        )
        
        # Verify versions
        assert len(versions) == 3, f"Expected 3 versions, got {len(versions)}"
        assert versions[0]["version"] == "3", "First version should be 3 (latest)"
        assert versions[1]["version"] == "2", "Second version should be 2"
        assert versions[2]["version"] == "1", "Third version should be 1 (oldest)"
        
    def test_list_model_versions_with_stages(self, mock_mlflow):
        """Test listing model versions filtered by stages."""
        # Configure mock for model versions
        client = mock_mlflow.tracking.MlflowClient.return_value
        
        # Create mock versions with different stages
        mock_versions = []
        stages = ["Production", "Staging", "Archived"]
        for i, stage in enumerate(stages, 1):
            mv = MagicMock()
            mv.version = str(i)
            mv.name = "test_model"
            mv.run_id = f"test_run_{i}"
            mv.current_stage = stage
            mv.creation_timestamp = 1000 * i
            mv.last_updated_timestamp = 1100 * i
            mv.description = f"Version {i}"
            mock_versions.append(mv)
        
        client.search_model_versions.return_value = mock_versions
        
        # List only Production and Staging models
        versions = list_model_versions(
            name="test_model",
            stages=["Production", "Staging"]
        )
        
        # Verify filtering
        assert len(versions) == 2, f"Expected 2 versions, got {len(versions)}"
        assert versions[0]["stage"] == "Staging", "First version should be Staging"
        assert versions[1]["stage"] == "Production", "Second version should be Production" 