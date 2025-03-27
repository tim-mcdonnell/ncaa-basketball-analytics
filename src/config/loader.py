"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        file_path: Path to the YAML configuration file

    Returns:
        Configuration as a dictionary

    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the configuration file is invalid YAML
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file {file_path}: {e}") from e


def merge_config_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.

    The override dictionary takes precedence over the base dictionary.
    This function performs a deep merge, preserving nested structures.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_config_dicts(result[key], value)
        else:
            # Override or add values
            result[key] = value

    return result


def load_config_with_overrides(
    base_path: Union[str, Path], override_paths: Optional[list[Union[str, Path]]] = None
) -> Dict[str, Any]:
    """Load a base configuration with optional overrides.

    Args:
        base_path: Path to the base configuration file
        override_paths: List of paths to override configuration files

    Returns:
        Merged configuration dictionary

    Raises:
        FileNotFoundError: If the base configuration file does not exist
    """
    config = load_config(base_path)

    if override_paths:
        for path in override_paths:
            if Path(path).exists():
                override = load_config(path)
                config = merge_config_dicts(config, override)

    return config


def apply_environment_variable_overrides(
    config: Dict[str, Any], prefix: str = "CONFIG_"
) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Environment variables with names like CONFIG_SECTION_KEY=value
    will override config["section"]["key"] = value.

    Args:
        config: Configuration dictionary to override
        prefix: Prefix for environment variables to consider

    Returns:
        Configuration dictionary with environment variable overrides applied
    """
    result = config.copy()

    for env_var, value in os.environ.items():
        if not env_var.startswith(prefix):
            continue

        # Remove prefix and split into parts
        config_path = env_var[len(prefix) :].lower().split("_")

        # Navigate to the nested dictionary
        current = result
        for part in config_path[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # If the path exists but is not a dict, convert it to a dict
                current[part] = {}
            current = current[part]

        # Set the value, converting to appropriate type if possible
        if value.lower() == "true":
            current[config_path[-1]] = True
        elif value.lower() == "false":
            current[config_path[-1]] = False
        elif value.isdigit():
            current[config_path[-1]] = int(value)
        elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
            current[config_path[-1]] = float(value)
        else:
            current[config_path[-1]] = value

    return result
