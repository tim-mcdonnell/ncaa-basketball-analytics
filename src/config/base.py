"""Base configuration classes and utilities."""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, create_model


class DotDict(Dict[str, Any]):
    """Dictionary subclass that allows dot notation access to nested dicts."""

    def __getattr__(self, key: str) -> Any:
        """Access dictionary keys as attributes."""
        if key in self:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                # Convert nested dictionaries to DotDict for recursive dot access
                self[key] = DotDict(value)
                return self[key]
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """Set dictionary keys as attributes."""
        self[key] = value

    def __delattr__(self, key: str) -> None:
        """Delete dictionary keys as attributes."""
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert DotDict back to a regular dictionary."""
        result: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class BaseConfig(BaseModel):
    """Base configuration class with enhanced functionality.

    All configuration models should inherit from this class to ensure
    consistent behavior and features across the configuration system.
    """

    model_config = ConfigDict(
        extra="forbid",  # Disallow extra fields not defined in the model
        validate_assignment=True,  # Validate values when setting attributes
        frozen=False,  # Allow modification after creation
        populate_by_name=True,  # Allow population by field name as well as alias
    )

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary, maintaining backward compatibility with Pydantic v1."""
        return self.model_dump(*args, **kwargs)

    def dot_dict(self) -> DotDict:
        """Convert to a dictionary that supports dot notation access."""
        return DotDict(self.dict())


def create_config_model(
    name: str, fields: Dict[str, Any], base_class: type = BaseConfig, **kwargs
) -> type:
    """Dynamically create a configuration model from a dictionary.

    Args:
        name: Name of the model class to create
        fields: Dictionary of field names and their types
        base_class: Base class to inherit from (defaults to BaseConfig)
        **kwargs: Additional arguments to pass to create_model

    Returns:
        A dynamically created Pydantic model class
    """
    return create_model(name, __base__=base_class, **fields, **kwargs)
