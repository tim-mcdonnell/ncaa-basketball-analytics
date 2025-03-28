"""
Configuration loading utilities.
"""

import yaml
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar


T = TypeVar("T", bound=BaseModel)


def load_config(config_path: str, config_model: Type[T]) -> T:
    """
    Loads and validates configuration from a YAML file using a Pydantic model.

    Args:
        config_path: Path to the YAML configuration file.
        config_model: The Pydantic model class to validate against.

    Returns:
        An instance of the Pydantic model populated with configuration data.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValidationError: If the configuration data fails validation.
        yaml.YAMLError: If the file is not valid YAML.
    """
    full_path = Path(config_path)
    if not full_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {full_path}")

    try:
        with open(full_path, "r") as f:
            config_data = yaml.safe_load(f)
            if config_data is None:
                config_data = {}  # Handle empty YAML file
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {full_path}: {e}")

    try:
        return config_model(**config_data)
    except ValidationError as e:
        raise ValidationError(
            f"Configuration validation failed for {full_path}: {e}", model=config_model
        )
