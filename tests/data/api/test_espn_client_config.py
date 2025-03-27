"""Tests for ESPN API client configuration loading."""

import pytest
from unittest.mock import patch, mock_open

from src.data.api.espn_client.config import ESPNConfig, load_espn_config


class TestESPNConfig:
    """Tests for ESPNConfig model."""

    def test_espn_config_defaults(self):
        """Test default configuration values."""
        config = ESPNConfig()

        assert (
            config.base_url
            == "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        )
        assert config.timeout == 30.0
        assert config.retries.max_attempts == 3
        assert config.retries.min_wait == 1.0
        assert config.retries.max_wait == 10.0
        assert config.retries.factor == 2.0
        assert config.rate_limiting.initial == 10
        assert config.rate_limiting.min_limit == 1
        assert config.rate_limiting.max_limit == 50
        assert config.metadata.dir == "data/metadata"
        assert config.metadata.file == "espn_metadata.json"

    def test_espn_config_custom_values(self):
        """Test custom configuration values."""
        config = ESPNConfig(
            base_url="https://custom-espn.com",
            timeout=60.0,
            retries={"max_attempts": 5, "min_wait": 2.0, "max_wait": 20.0, "factor": 3.0},
            rate_limiting={
                "initial": 20,
                "min_limit": 2,
                "max_limit": 100,
                "success_threshold": 20,
                "failure_threshold": 5,
            },
            metadata={"dir": "custom/metadata", "file": "custom.json"},
        )

        assert config.base_url == "https://custom-espn.com"
        assert config.timeout == 60.0
        assert config.retries.max_attempts == 5
        assert config.retries.min_wait == 2.0
        assert config.retries.max_wait == 20.0
        assert config.retries.factor == 3.0
        assert config.rate_limiting.initial == 20
        assert config.rate_limiting.min_limit == 2
        assert config.rate_limiting.max_limit == 100
        assert config.rate_limiting.success_threshold == 20
        assert config.rate_limiting.failure_threshold == 5
        assert config.metadata.dir == "custom/metadata"
        assert config.metadata.file == "custom.json"


class TestESPNConfigLoading:
    """Tests for configuration loading from YAML."""

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_espn_config_from_yaml(self, mock_yaml_load, mock_file_open, mock_path_exists):
        """Test loading configuration from YAML file."""
        # Mock file exists check
        mock_path_exists.return_value = True

        # Mock YAML config data
        mock_yaml_data = {
            "espn": {
                "base_url": "https://test-espn.com",
                "timeout": 45.0,
                "retries": {"max_attempts": 4},
                "rate_limiting": {"initial": 15},
            }
        }
        mock_yaml_load.return_value = mock_yaml_data

        # Load config
        config = load_espn_config("test/config.yaml")

        # Verify config
        assert config.base_url == "https://test-espn.com"
        assert config.timeout == 45.0
        assert config.retries.max_attempts == 4
        assert config.rate_limiting.initial == 15

        # Verify other values still have defaults
        assert config.retries.min_wait == 1.0
        assert config.metadata.dir == "data/metadata"

    @patch("os.path.exists")
    def test_load_espn_config_file_not_found(self, mock_path_exists):
        """Test loading default configuration when file not found."""
        # Mock file exists check
        mock_path_exists.return_value = False

        # Load config (should use defaults)
        config = load_espn_config("nonexistent/config.yaml")

        # Verify defaults
        assert (
            config.base_url
            == "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        )
        assert config.timeout == 30.0

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_espn_config_invalid_yaml(self, mock_yaml_load, mock_file_open, mock_path_exists):
        """Test handling of invalid YAML configuration."""
        # Mock file exists check
        mock_path_exists.return_value = True

        # Mock YAML load error
        mock_yaml_load.side_effect = Exception("Invalid YAML")

        # Load config should raise exception
        with pytest.raises(ValueError) as exc_info:
            load_espn_config("invalid/config.yaml")

        # Verify error message
        assert "Invalid ESPN config" in str(exc_info.value)
