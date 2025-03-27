"""Tests for configuration loading functionality."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_config, merge_config_dicts, apply_environment_variable_overrides


def test_load_config():
    """Test loading configuration from YAML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"

        # Create a test configuration
        test_config = {
            "database": {"host": "localhost", "port": 5432, "username": "user", "password": "pass"}
        }

        # Write configuration to file
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)

        # Test loading the configuration
        loaded_config = load_config(config_path)
        assert loaded_config == test_config

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            load_config(Path(tmpdir) / "nonexistent.yaml")


def test_merge_config_dicts():
    """Test merging of configuration dictionaries."""
    # Base configuration
    base_config = {
        "server": {
            "host": "localhost",
            "port": 8080,
            "debug": False,
            "timeout": 30,
            "options": {"keepalive": True, "workers": 4},
        },
        "logging": {"level": "INFO", "format": "standard"},
    }

    # Override configuration
    override_config = {
        "server": {"port": 9090, "debug": True, "options": {"workers": 8}},
        "database": {"url": "postgres://localhost/db"},
    }

    # Expected merged result
    expected_result = {
        "server": {
            "host": "localhost",  # From base
            "port": 9090,  # Overridden
            "debug": True,  # Overridden
            "timeout": 30,  # From base
            "options": {
                "keepalive": True,  # From base
                "workers": 8,  # Overridden
            },
        },
        "logging": {"level": "INFO", "format": "standard"},
        "database": {
            "url": "postgres://localhost/db"  # New in override
        },
    }

    # Test merging
    result = merge_config_dicts(base_config, override_config)
    assert result == expected_result

    # Original dictionaries should not be modified
    assert base_config["server"]["port"] == 8080
    assert "database" not in base_config


def test_config_overrides():
    """Test precedence of configuration overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create base configuration
        base_config = {
            "application": {
                "name": "TestApp",
                "version": "1.0.0",
                "log_level": "INFO",
                "features": {"feature1": True, "feature2": False, "feature3": True},
            }
        }

        # Create environment-specific configuration
        env_config = {"application": {"log_level": "DEBUG", "features": {"feature2": True}}}

        # Create instance-specific configuration
        instance_config = {"application": {"features": {"feature3": False}}}

        # Write configurations to files
        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(base_config, f)

        with open(config_dir / "config.development.yaml", "w") as f:
            yaml.dump(env_config, f)

        with open(config_dir / "config.instance.yaml", "w") as f:
            yaml.dump(instance_config, f)

        # Set environment variables for final overrides
        os.environ["CONFIG_APPLICATION_FEATURES_FEATURE1"] = "false"

        # Expected result after all overrides
        expected_config = {
            "application": {
                "name": "TestApp",  # From base
                "version": "1.0.0",  # From base
                "log_level": "DEBUG",  # From env
                "features": {
                    "feature1": False,  # From env var
                    "feature2": True,  # From env
                    "feature3": False,  # From instance
                },
            }
        }

        # Load the base configuration
        merged = load_config(config_dir / "config.yaml")

        # Merge with environment-specific configuration
        env_merged = load_config(config_dir / "config.development.yaml")
        merged = merge_config_dicts(merged, env_merged)

        # Merge with instance-specific configuration
        instance_merged = load_config(config_dir / "config.instance.yaml")
        merged = merge_config_dicts(merged, instance_merged)

        # Apply environment variable overrides
        merged = apply_environment_variable_overrides(merged)

        # Test the override precedence
        assert merged == expected_config

        # Clean up
        os.environ.pop("CONFIG_APPLICATION_FEATURES_FEATURE1", None)
