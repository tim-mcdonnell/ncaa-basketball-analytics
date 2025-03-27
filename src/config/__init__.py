"""Configuration management system for NCAA Basketball Analytics.

This module provides a robust configuration system for managing settings
across different components and environments. It supports:

- Hierarchical configuration with inheritance
- Environment-specific configuration (development, testing, production)
- Type validation using Pydantic models
- Default values for optional settings
- Configuration override from environment variables
- Versioning and compatibility checking
"""

from src.config.base import BaseConfig, DotDict, create_config_model
from src.config.environment import get_environment, load_environment_config
from src.config.loader import (
    apply_environment_variable_overrides,
    load_config,
    load_config_with_overrides,
    merge_config_dicts,
)
from src.config.models import (
    ApiConfig,
    DbConfig,
    FeatureConfig,
    ModelConfig,
    DashboardConfig,
)
from src.config.settings import get_default_settings, ConfigSchema
from src.config.validation import validate_config, validate_and_convert
from src.config.versioning import check_config_version, ConfigVersionError

# Current version of the configuration system
__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseConfig",
    "DotDict",
    "create_config_model",
    # Environment handling
    "get_environment",
    "load_environment_config",
    # Loading utilities
    "load_config",
    "load_config_with_overrides",
    "merge_config_dicts",
    "apply_environment_variable_overrides",
    # Validation utilities
    "validate_config",
    "validate_and_convert",
    # Versioning utilities
    "check_config_version",
    "ConfigVersionError",
    # Configuration models
    "ApiConfig",
    "DbConfig",
    "FeatureConfig",
    "ModelConfig",
    "DashboardConfig",
    # Settings
    "get_default_settings",
    "ConfigSchema",
]
