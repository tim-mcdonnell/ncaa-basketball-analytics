"""Default configuration settings and schemas."""

from src.config.settings.defaults import get_default_settings
from src.config.settings.schema import ConfigSchema

__all__ = ["get_default_settings", "ConfigSchema"]
