"""Tests for environment-specific configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.environment import get_environment, load_environment_config
from src.config.models.api_config import ApiConfig


def test_get_environment():
    """Test retrieving the current environment name."""
    # Test default environment
    os.environ.pop("ENV", None)  # Clear ENV if set
    assert get_environment() == "development"  # Default should be development

    # Test setting environment via environment variable
    os.environ["ENV"] = "production"
    assert get_environment() == "production"

    # Test setting environment via environment variable (case insensitive)
    os.environ["ENV"] = "TESTING"
    assert get_environment() == "testing"

    # Reset for other tests
    os.environ.pop("ENV", None)


def test_environment_loading():
    """Test loading configuration based on the environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        # Create a base config
        base_config = {"api": {"host": "localhost", "port": 8000, "debug": False, "timeout": 30}}

        # Create environment-specific configs
        dev_config = {"api": {"debug": True, "timeout": 60}}

        prod_config = {"api": {"host": "api.example.com", "port": 443, "timeout": 10}}

        # Write configs to files
        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(base_config, f)

        with open(config_dir / "config.development.yaml", "w") as f:
            yaml.dump(dev_config, f)

        with open(config_dir / "config.production.yaml", "w") as f:
            yaml.dump(prod_config, f)

        # Test development environment
        os.environ["ENV"] = "development"
        config = load_environment_config(ApiConfig, config_dir)
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.debug is True  # Overridden in dev config
        assert config.timeout == 60  # Overridden in dev config

        # Test production environment
        os.environ["ENV"] = "production"
        config = load_environment_config(ApiConfig, config_dir)
        assert config.host == "api.example.com"
        assert config.port == 443
        assert config.debug is False  # Not overridden in prod config
        assert config.timeout == 10  # Overridden in prod config

        # Reset for other tests
        os.environ.pop("ENV", None)


def test_environment_variable_override():
    """Test environment variable overrides for configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        # Create a base config
        base_config = {"api": {"host": "localhost", "port": 8000, "debug": False, "timeout": 30}}

        # Write config to file
        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(base_config, f)

        # Set environment variables to override configuration
        os.environ["CONFIG_API_HOST"] = "override.example.com"
        os.environ["CONFIG_API_PORT"] = "9000"

        # Test that environment variables override file settings
        config = load_environment_config(ApiConfig, config_dir)
        assert config.host == "override.example.com"  # From env var
        assert config.port == 9000  # From env var (converted to int)
        assert config.debug is False  # From file
        assert config.timeout == 30  # From file

        # Clean up environment variables
        os.environ.pop("CONFIG_API_HOST", None)
        os.environ.pop("CONFIG_API_PORT", None)
        os.environ.pop("ENV", None)


def test_config_missing_files():
    """Test graceful handling of missing configuration files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        # No files at all - should raise an error
        with pytest.raises(FileNotFoundError):
            load_environment_config(ApiConfig, config_dir)

        # Create only base config
        base_config = {"api": {"host": "localhost", "port": 8000, "debug": False, "timeout": 30}}

        # Write config to file
        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(base_config, f)

        # Set to an environment with no specific config file
        os.environ["ENV"] = "testing"

        # Should load only base config without errors
        config = load_environment_config(ApiConfig, config_dir)
        assert config.host == "localhost"
        assert config.port == 8000

        # Clean up
        os.environ.pop("ENV", None)
