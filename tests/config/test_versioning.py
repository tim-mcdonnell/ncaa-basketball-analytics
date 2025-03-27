"""Tests for configuration versioning functionality."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.versioning import check_config_version, ConfigVersionError, migrate_config


def test_config_version_compatibility():
    """Test version compatibility checking for configuration files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        # Create a test configuration with version
        test_config = {"_version": "1.0.0", "database": {"host": "localhost", "port": 5432}}

        # Write configuration to file
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)

        # Test compatible version - exact match
        assert check_config_version(config_path, "1.0.0") is True

        # Test compatible version - compatible minor version
        assert check_config_version(config_path, "1.1.0") is True

        # Test incompatible version - major version change
        with pytest.raises(ConfigVersionError):
            check_config_version(config_path, "2.0.0")

        # Test missing version
        with open(config_path, "w") as f:
            yaml.dump({"database": {"host": "localhost"}}, f)

        with pytest.raises(ConfigVersionError):
            check_config_version(config_path, "1.0.0")


def test_version_migration():
    """Test configuration version migration handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        # Create a test configuration with old version
        old_config = {
            "_version": "1.0.0",
            "database": {"connection_string": "postgresql://user:pass@localhost/db"},
        }

        # Write old configuration to file
        with open(config_path, "w") as f:
            yaml.dump(old_config, f)

        # Call the migration function
        migrated_config = migrate_config(config_path, target_version="2.0.0")

        # Verify migration results
        assert migrated_config["_version"] == "2.0.0"
        assert migrated_config["database"]["host"] == "localhost"
        assert migrated_config["database"]["port"] == 5432
        assert migrated_config["database"]["username"] == "user"
        assert migrated_config["database"]["password"] == "pass"
        assert migrated_config["database"]["database"] == "db"

        # Test with unsupported migration
        with open(config_path, "w") as f:
            yaml.dump({"_version": "1.0.0"}, f)

        with pytest.raises(ConfigVersionError):
            migrate_config(config_path, target_version="3.0.0")
