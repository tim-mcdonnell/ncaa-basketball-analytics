"""Configuration schema definitions."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ConfigSchema(BaseModel):
    """Configuration schema model for validation and documentation.

    This model defines the structure and metadata for the configuration schema.
    It is used for generating documentation and validating configuration files.
    """

    name: str = Field(description="Name of the configuration schema")
    version: str = Field(description="Version of the configuration schema (X.Y.Z format)")
    description: str = Field(description="Description of the configuration schema")
    components: Dict[str, Dict] = Field(description="Components defined in the schema")
    required_components: List[str] = Field(description="List of required components")
    examples: Optional[Dict[str, str]] = Field(default=None, description="Example configurations")


# Default schema for NCAA Basketball Analytics
default_schema = ConfigSchema(
    name="NCAA Basketball Analytics Configuration",
    version="1.0.0",
    description="Configuration schema for NCAA Basketball Analytics project",
    components={
        "api": {
            "description": "API server configuration",
            "properties": ["host", "port", "debug", "timeout", "rate_limit", "endpoints"],
        },
        "db": {
            "description": "Database configuration",
            "properties": [
                "path",
                "read_only",
                "memory_map",
                "threads",
                "extensions",
                "allow_external_access",
                "cache_size",
            ],
        },
        "feature": {
            "description": "Feature calculation configuration",
            "properties": [
                "enabled",
                "cache_results",
                "parameters",
                "dependencies",
                "priority",
                "max_cache_age_seconds",
            ],
        },
        "model": {
            "description": "Machine learning model configuration",
            "properties": [
                "type",
                "feature_set",
                "target",
                "hyperparameters",
                "evaluation",
                "experiment_tracking",
                "random_seed",
            ],
        },
        "dashboard": {
            "description": "Dashboard and visualization configuration",
            "properties": [
                "title",
                "refresh_interval",
                "theme",
                "layout",
                "default_view",
                "available_views",
                "cache_timeout",
                "show_filters",
                "max_items_per_page",
            ],
        },
    },
    required_components=["api", "db"],
    examples={
        "minimal": """
        api:
          host: localhost
          port: 8000
        db:
          path: ./data/basketball.duckdb
        """,
        "development": """
        api:
          host: localhost
          port: 8000
          debug: true
        db:
          path: ./data/basketball.duckdb
          read_only: false
        feature:
          enabled: true
          cache_results: true
        """,
    },
)
