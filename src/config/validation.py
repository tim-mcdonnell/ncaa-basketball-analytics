"""Configuration validation utilities."""

from typing import Any, Dict, Type, TypeVar

from pydantic import ValidationError

from src.config.base import BaseConfig, DotDict

T = TypeVar("T", bound=BaseConfig)


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors.

    This exception wraps Pydantic's ValidationError and provides
    more context and clear error messages specific to configuration.
    """

    def __init__(self, message: str, validation_error: ValidationError):
        """Initialize with message and original validation error.

        Args:
            message: User-friendly error message
            validation_error: Original Pydantic ValidationError
        """
        self.validation_error = validation_error
        self.errors = validation_error.errors()
        super().__init__(f"{message}: {validation_error}")


def validate_config(config_model: Type[T], config_data: Dict[str, Any]) -> T:
    """Validate configuration data against a configuration model.

    Args:
        config_model: Configuration model class to validate against
        config_data: Configuration data to validate

    Returns:
        Validated configuration model instance

    Raises:
        ValidationError: If validation fails due to Pydantic validation
    """
    # We don't wrap the ValidationError here to make testing easier
    return config_model(**config_data)


def validate_and_convert(config_model: Type[T], config_data: Dict[str, Any]) -> DotDict:
    """Validate and convert configuration to dot notation format.

    This is a convenience function that combines validation and
    conversion to dot notation access in one step.

    Args:
        config_model: Configuration model class to validate against
        config_data: Configuration data to validate

    Returns:
        Validated configuration as a DotDict for dot notation access

    Raises:
        ValidationError: If validation fails
    """
    model = validate_config(config_model, config_data)
    return model.dot_dict()
