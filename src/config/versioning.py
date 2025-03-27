"""Configuration versioning utilities."""

import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from src.config.loader import load_config


class ConfigVersionError(Exception):
    """Exception raised for configuration version errors."""

    pass


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse a version string into a tuple of (major, minor, patch).

    Args:
        version_str: Version string in the format 'X.Y.Z'

    Returns:
        Tuple of (major, minor, patch) version components

    Raises:
        ValueError: If the version string is invalid
    """
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}. Expected format: X.Y.Z")

    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def is_version_compatible(required_version: str, current_version: str) -> bool:
    """Check if a version is compatible with a required version.

    Two versions are considered compatible if they have the same major version.

    Args:
        required_version: Required version string
        current_version: Current version string

    Returns:
        True if versions are compatible, False otherwise

    Raises:
        ValueError: If either version string is invalid
    """
    required_major, _, _ = parse_version(required_version)
    current_major, _, _ = parse_version(current_version)

    return required_major == current_major


def get_config_version(config_path: Union[str, Path]) -> Optional[str]:
    """Get the version of a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Version string, or None if no version is specified

    Raises:
        FileNotFoundError: If the configuration file does not exist
    """
    config = load_config(config_path)
    return config.get("_version")


def check_config_version(config_path: Union[str, Path], required_version: str) -> bool:
    """Check if a configuration file version is compatible with a required version.

    Args:
        config_path: Path to the configuration file
        required_version: Required version string

    Returns:
        True if versions are compatible

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ConfigVersionError: If the version is missing or incompatible
    """
    config_version = get_config_version(config_path)

    if config_version is None:
        raise ConfigVersionError(f"Configuration file {config_path} does not specify a version.")

    if not is_version_compatible(required_version, config_version):
        raise ConfigVersionError(
            f"Configuration file {config_path} has incompatible version {config_version}. "
            f"Required version: {required_version}."
        )

    return True


def migrate_config(
    config_path: Union[str, Path], target_version: str, save: bool = False
) -> Dict[str, Any]:
    """Migrate a configuration file to a new version.

    This function performs schema migrations between configuration versions.
    Currently supports migrating from v1.0.0 to v2.0.0 for database configuration.

    Args:
        config_path: Path to the configuration file
        target_version: Target version to migrate to
        save: Whether to save the migrated configuration back to the file

    Returns:
        Migrated configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ConfigVersionError: If migration is not supported for this version
    """
    config = load_config(config_path)
    current_version = config.get("_version")

    if not current_version:
        raise ConfigVersionError(f"Configuration file {config_path} does not specify a version.")

    # Implement migration logic
    if current_version == "1.0.0" and target_version == "2.0.0":
        # Specifically migrate database section from v1.0.0 to v2.0.0
        if "database" in config and "connection_string" in config["database"]:
            # Parse connection string
            conn_str = config["database"]["connection_string"]
            conn_parts = {}

            # Simple connection string parser
            if "://" in conn_str:
                userpass, hostport_db = conn_str.split("@", 1)
                protocol_userpass = userpass.split("://", 1)
                if len(protocol_userpass) > 1:
                    _, userpass = protocol_userpass
                else:
                    userpass = protocol_userpass[0]

                if ":" in userpass:
                    conn_parts["username"], conn_parts["password"] = userpass.split(":", 1)
                else:
                    conn_parts["username"] = userpass

                if "/" in hostport_db:
                    hostport, db = hostport_db.split("/", 1)
                    conn_parts["database"] = db

                    if ":" in hostport:
                        conn_parts["host"], port = hostport.split(":", 1)
                        conn_parts["port"] = int(port)
                    else:
                        conn_parts["host"] = hostport
                else:
                    conn_parts["host"] = hostport_db

            # Create new database configuration
            new_db_config = {
                "host": conn_parts.get("host", "localhost"),
                "port": conn_parts.get("port", 5432),
                "username": conn_parts.get("username", ""),
                "password": conn_parts.get("password", ""),
                "database": conn_parts.get("database", ""),
            }

            # Update the configuration
            config["database"] = new_db_config
            config["_version"] = target_version

            # Save if requested
            if save:
                with open(config_path, "w") as f:
                    yaml.dump(config, f)

            return config
        else:
            # Update version even if there's no database section to migrate
            config["_version"] = target_version

            if save:
                with open(config_path, "w") as f:
                    yaml.dump(config, f)

            return config
    else:
        raise ConfigVersionError(
            f"Migration from version {current_version} to {target_version} is not supported."
        )

    return config
