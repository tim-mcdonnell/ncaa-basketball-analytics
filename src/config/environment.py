"""Environment-specific configuration handling."""

import os
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

from src.config.base import BaseConfig
from src.config.loader import (
    apply_environment_variable_overrides,
    load_config,
    merge_config_dicts,
)
from src.config.validation import validate_config

# Valid environment names
VALID_ENVIRONMENTS = ["development", "testing", "production"]
DEFAULT_ENVIRONMENT = "development"

T = TypeVar("T", bound=BaseConfig)


def get_environment() -> str:
    """Get the current environment name.

    The environment is determined by the ENV environment variable.
    If not set, it defaults to "development".

    Returns:
        Environment name (lowercase)
    """
    env = os.environ.get("ENV", DEFAULT_ENVIRONMENT).lower()
    return env


def is_valid_environment(env: str) -> bool:
    """Check if an environment name is valid.

    Args:
        env: Environment name to check

    Returns:
        True if the environment is valid, False otherwise
    """
    return env.lower() in VALID_ENVIRONMENTS


def get_environment_config_path(
    config_dir: Union[str, Path], env: Optional[str] = None, base_name: str = "config"
) -> Path:
    """Get the path to an environment-specific configuration file.

    Args:
        config_dir: Directory containing configuration files
        env: Environment name (defaults to current environment)
        base_name: Base name of the configuration file

    Returns:
        Path to the environment-specific configuration file
    """
    config_dir = Path(config_dir)
    env = env or get_environment()

    return config_dir / f"{base_name}.{env}.yaml"


def load_environment_config(
    config_model: Type[T],
    config_dir: Union[str, Path],
    env: Optional[str] = None,
    base_name: str = "config",
    apply_env_vars: bool = True,
) -> T:
    """Load environment-specific configuration.

    This function loads the base configuration and overlays the
    environment-specific configuration on top of it. It then validates
    the resulting configuration against the specified model.

    Args:
        config_model: Configuration model class to validate against
        config_dir: Directory containing configuration files
        env: Environment name (defaults to current environment)
        base_name: Base name of the configuration file
        apply_env_vars: Whether to apply environment variable overrides

    Returns:
        Validated configuration model instance

    Raises:
        FileNotFoundError: If the base configuration file does not exist
    """
    config_dir = Path(config_dir)
    env = env or get_environment()

    # Base configuration file
    base_config_path = config_dir / f"{base_name}.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")

    # Load base configuration
    config = load_config(base_config_path)

    # Load environment-specific configuration if it exists
    env_config_path = get_environment_config_path(config_dir, env, base_name)
    if env_config_path.exists():
        env_config = load_config(env_config_path)
        config = merge_config_dicts(config, env_config)

    # Apply environment variable overrides if requested
    if apply_env_vars:
        config = apply_environment_variable_overrides(config)

    # Extract the relevant section based on the model name
    # This assumes that the model name without "Config" suffix is the section name
    section_name = config_model.__name__.lower().replace("config", "")
    section_data = config.get(section_name, {})

    # Validate against the model
    return validate_config(config_model, section_data)
