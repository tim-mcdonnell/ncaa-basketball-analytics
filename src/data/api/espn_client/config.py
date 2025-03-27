"""ESPN API configuration loading."""

import os
from typing import Optional
import logging
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetryConfig(BaseModel):
    """Configuration for API request retries."""

    max_attempts: int = Field(default=3, description="Maximum number of retry attempts")
    min_wait: float = Field(default=1.0, description="Minimum wait time between retries (seconds)")
    max_wait: float = Field(default=10.0, description="Maximum wait time between retries (seconds)")
    factor: float = Field(default=2.0, description="Exponential backoff factor")


class RateLimitConfig(BaseModel):
    """Configuration for adaptive rate limiting."""

    initial: int = Field(default=10, description="Initial concurrency limit")
    min_limit: int = Field(default=1, description="Minimum concurrency limit")
    max_limit: int = Field(default=50, description="Maximum concurrency limit")
    success_threshold: int = Field(default=10, description="Success threshold for limit increase")
    failure_threshold: int = Field(default=3, description="Failure threshold for limit decrease")


class MetadataConfig(BaseModel):
    """Configuration for metadata storage."""

    dir: str = Field(default="data/metadata", description="Directory for metadata storage")
    file: str = Field(default="espn_metadata.json", description="Filename for metadata storage")


class ESPNConfig(BaseModel):
    """ESPN API configuration model."""

    base_url: str = Field(
        default="https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball",
        description="Base URL for ESPN API requests",
    )
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    retries: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")
    rate_limiting: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting configuration"
    )
    metadata: MetadataConfig = Field(
        default_factory=MetadataConfig, description="Metadata configuration"
    )


def load_espn_config(config_path: Optional[str] = None) -> ESPNConfig:
    """
    Load ESPN API configuration from a YAML file.

    Args:
        config_path: Path to config file (default looks in config/api/espn.yaml)

    Returns:
        ESPNConfig: Parsed configuration

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    """
    # Default config path
    if not config_path:
        config_path = os.path.join("config", "api", "espn.yaml")

    try:
        # Check if file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return ESPNConfig()

        # Load YAML
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Extract ESPN section
        espn_config = config_data.get("espn", {})

        # Parse with pydantic
        return ESPNConfig(**espn_config)
    except Exception as e:
        logger.error(f"Error loading ESPN config: {e}")
        raise ValueError(f"Invalid ESPN config: {e}") from e
