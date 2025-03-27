"""Tests for API configuration model."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config.models.api_config import ApiConfig


def test_api_config_validation():
    """Test validation of API configuration."""
    # Valid configuration
    valid_config = {
        "host": "localhost",
        "port": 8080,
        "debug": True,
        "timeout": 30,
        "rate_limit": 100,
        "endpoints": {"teams": "/api/teams", "games": "/api/games"},
    }

    api_config = ApiConfig(**valid_config)
    assert api_config.host == "localhost"
    assert api_config.port == 8080
    assert api_config.debug is True
    assert api_config.timeout == 30
    assert api_config.rate_limit == 100
    assert api_config.endpoints.teams == "/api/teams"
    assert api_config.endpoints.games == "/api/games"


def test_api_config_validation_errors():
    """Test validation errors for invalid API configuration."""
    # Invalid port (out of range)
    invalid_port_config = {
        "host": "localhost",
        "port": 70000,  # Invalid port number
        "debug": True,
        "timeout": 30,
    }
    with pytest.raises(ValidationError) as excinfo:
        ApiConfig(**invalid_port_config)
    error_text = str(excinfo.value)
    assert "port" in error_text
    assert "65535" in error_text  # Max port value

    # Invalid timeout (negative)
    invalid_timeout_config = {
        "host": "localhost",
        "port": 8080,
        "debug": True,
        "timeout": -5,  # Invalid timeout
    }
    with pytest.raises(ValidationError) as excinfo:
        ApiConfig(**invalid_timeout_config)
    error_text = str(excinfo.value)
    assert "timeout" in error_text
    assert "greater than or equal to 0" in error_text


def test_api_config_defaults():
    """Test default values for API configuration."""
    # Minimal configuration
    minimal_config = {"host": "localhost", "port": 8080}

    api_config = ApiConfig(**minimal_config)
    assert api_config.host == "localhost"
    assert api_config.port == 8080
    assert api_config.debug is False  # Default
    assert api_config.timeout == 60  # Default
    assert api_config.rate_limit == 0  # Default (unlimited)
    assert hasattr(api_config, "endpoints")  # Should have endpoints field


def test_api_config_from_yaml():
    """Test loading API configuration from YAML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "api_config.yaml"

        # Create a test configuration
        test_config = {
            "api": {
                "host": "api.example.com",
                "port": 443,
                "debug": False,
                "timeout": 15,
                "rate_limit": 1000,
                "endpoints": {"teams": "/v1/teams", "games": "/v1/games", "stats": "/v1/stats"},
            }
        }

        # Write configuration to file
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)

        # Load the configuration
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)

        # Create ApiConfig from loaded data
        api_config = ApiConfig(**loaded_config["api"])

        # Verify configuration
        assert api_config.host == "api.example.com"
        assert api_config.port == 443
        assert api_config.debug is False
        assert api_config.timeout == 15
        assert api_config.rate_limit == 1000
        assert api_config.endpoints.teams == "/v1/teams"
        assert api_config.endpoints.games == "/v1/games"
        assert api_config.endpoints.stats == "/v1/stats"
