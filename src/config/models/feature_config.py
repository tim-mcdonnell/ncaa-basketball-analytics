"""Feature configuration model."""

from typing import Dict, List, Optional, Union

from pydantic import Field

from src.config.base import BaseConfig


class FeatureParameter(BaseConfig):
    """Parameter for a feature configuration.

    This model defines a parameter that can be used in a feature calculation.

    Example YAML configuration:
    ```yaml
    feature:
      parameters:
        window_size:
          value: 5
          description: "Window size for rolling calculations"
          min: 1
          max: 20
    ```
    """

    value: Union[int, float, str, bool, List[str]] = Field(description="Value of the parameter")
    description: Optional[str] = Field(default=None, description="Description of the parameter")
    min: Optional[Union[int, float]] = Field(
        default=None, description="Minimum allowed value (for numeric parameters)"
    )
    max: Optional[Union[int, float]] = Field(
        default=None, description="Maximum allowed value (for numeric parameters)"
    )
    options: Optional[List[str]] = Field(
        default=None, description="List of allowed options (for string parameters)"
    )


class FeatureConfig(BaseConfig):
    """Feature configuration model.

    This model defines the configuration for features and calculations.

    Example YAML configuration:
    ```yaml
    feature:
      enabled: true
      cache_results: true
      parameters:
        smoothing_factor:
          value: 0.3
          description: "Smoothing factor for EMA calculation"
          min: 0.0
          max: 1.0
        metric_type:
          value: "efficiency"
          description: "Type of efficiency metric to use"
          options: ["raw", "efficiency", "adjusted"]
      dependencies:
        - stats
        - player_data
    ```
    """

    enabled: bool = Field(default=True, description="Whether the feature calculation is enabled")
    cache_results: bool = Field(default=True, description="Whether to cache calculation results")
    parameters: Dict[str, FeatureParameter] = Field(
        default_factory=dict, description="Parameters for the feature calculation"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="List of dependencies for this feature"
    )
    priority: int = Field(
        default=100, description="Priority for feature calculation order (lower runs first)", ge=0
    )
    max_cache_age_seconds: Optional[int] = Field(
        default=None,
        description="Maximum age of cached results in seconds (None for no limit)",
        ge=0,
    )
