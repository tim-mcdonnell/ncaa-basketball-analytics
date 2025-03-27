"""API configuration model."""

from typing import Optional

from pydantic import Field, field_validator

from src.config.base import BaseConfig


class ApiEndpoints(BaseConfig):
    """API endpoints configuration.

    This model defines the endpoints available in the API.

    Example YAML configuration:
    ```yaml
    api:
      endpoints:
        teams: /api/teams
        games: /api/games
        stats: /api/stats
    ```
    """

    teams: str = Field(default="/api/teams", description="Teams API endpoint path")
    games: str = Field(default="/api/games", description="Games API endpoint path")
    stats: str = Field(default="/api/stats", description="Stats API endpoint path")


class ApiConfig(BaseConfig):
    """API configuration model.

    This model defines the configuration for the API component.

    Example YAML configuration:
    ```yaml
    api:
      host: localhost
      port: 8080
      debug: true
      timeout: 30
      rate_limit: 100
      endpoints:
        teams: /api/teams
        games: /api/games
        stats: /api/stats
    ```
    """

    host: str = Field(
        description="API server hostname",
    )
    port: int = Field(description="API server port", ge=1, le=65535)
    debug: bool = Field(default=False, description="Enable debug mode for the API server")
    timeout: int = Field(default=60, description="Request timeout in seconds", ge=0)
    rate_limit: int = Field(
        default=0, description="Rate limit (requests per minute, 0 for unlimited)", ge=0
    )
    endpoints: Optional[ApiEndpoints] = Field(
        default_factory=ApiEndpoints, description="API endpoints configuration"
    )

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate that the host is not empty."""
        if not v:
            raise ValueError("Host cannot be empty")
        return v
