import pytest

from src.models.base import BaseModel, ModelConfig, ModelVersion


class TestBaseModel:
    """Test suite for BaseModel interface."""

    def test_model_interface(self):
        """Test that BaseModel interface defines required methods and properties."""
        # This test verifies the interface itself by checking that methods exist
        required_methods = [
            "__init__",
            "forward",
            "predict",
            "save",
            "load",
            "get_version",
            "get_hyperparameters",
        ]

        for method in required_methods:
            assert hasattr(BaseModel, method), f"BaseModel missing required method: {method}"

    def test_model_config(self):
        """Test that ModelConfig has required attributes."""
        required_attrs = [
            "model_type",
            "hyperparameters",
            "features",
            "training_params",
        ]

        for attr in required_attrs:
            assert hasattr(ModelConfig, attr), f"ModelConfig missing required attribute: {attr}"

    def test_model_version(self):
        """Test that ModelVersion has required attributes."""
        required_attrs = [
            "version_id",
            "creation_timestamp",
            "model_type",
            "hyperparameters",
            "features",
        ]

        for attr in required_attrs:
            assert hasattr(ModelVersion, attr), f"ModelVersion missing required attribute: {attr}"


class TestModelSerialization:
    """Test suite for model serialization."""

    def test_model_serialization(self, tmp_path):
        """Test that models can be saved and loaded."""
        # This is a skeleton test - actual implementation will create a concrete model
        # derived from BaseModel, save it, and load it
        pytest.skip("Implementation pending concrete model class")
