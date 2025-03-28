"""
Pydantic models for application configuration.
"""

from pydantic import BaseModel, Field
from typing import Optional


class DatabaseConfig(BaseModel):
    """Configuration for the database connection."""

    path: str = Field(
        ..., description="Path to the DuckDB database file relative to the project root."
    )


class APIConfig(BaseModel):
    """Configuration for external APIs."""

    # Example: Add API specific fields here
    # espn_api_key: Optional[str] = None
    pass


class FeatureConfig(BaseModel):
    """Configuration for feature generation."""

    # Example: Add feature config fields here
    # lookback_window: int = 30
    pass


class ModelTrainingConfig(BaseModel):
    """Configuration for model training."""

    # Example: Add training config fields here
    # default_epochs: int = 50
    # default_batch_size: int = 64
    pass


class AppSettings(BaseModel):
    """Main application settings model, potentially loading other configs."""

    database: DatabaseConfig
    api: Optional[APIConfig] = None
    features: Optional[FeatureConfig] = None
    training: Optional[ModelTrainingConfig] = None
