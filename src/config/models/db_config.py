"""Database configuration model."""

from typing import List, Optional

from pydantic import Field, field_validator

from src.config.base import BaseConfig


class DbConfig(BaseConfig):
    """Database configuration model.

    This model defines the configuration for the database component.

    Example YAML configuration:
    ```yaml
    db:
      path: ./data/basketball.duckdb
      read_only: false
      memory_map: true
      threads: 4
      extensions:
        - json
        - httpfs
      allow_external_access: false
    ```
    """

    path: str = Field(
        description="Path to the DuckDB database file",
    )
    read_only: bool = Field(default=False, description="Open the database in read-only mode")
    memory_map: bool = Field(default=True, description="Memory-map the database file")
    threads: int = Field(
        default=0, description="Number of threads to use (0 for system default)", ge=0
    )
    extensions: List[str] = Field(
        default_factory=list, description="List of DuckDB extensions to load"
    )
    allow_external_access: bool = Field(
        default=False, description="Allow database to access external resources (https, s3, etc.)"
    )
    cache_size: Optional[int] = Field(default=None, description="Database cache size in MB", ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate that the database path is not empty."""
        if not v:
            raise ValueError("Database path cannot be empty")
        return v


class ConnectionPoolConfig(BaseConfig):
    """Connection pool configuration.

    This model defines configuration for a database connection pool.

    Example YAML configuration:
    ```yaml
    db:
      connection_pool:
        min_connections: 5
        max_connections: 20
        connection_timeout: 30
    ```
    """

    min_connections: int = Field(
        default=1, description="Minimum number of connections in the pool", ge=1
    )
    max_connections: int = Field(
        default=10, description="Maximum number of connections in the pool", ge=1
    )
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds", ge=0)

    @field_validator("max_connections")
    @classmethod
    def validate_max_connections(cls, v: int, values) -> int:
        """Validate that max_connections is greater than or equal to min_connections."""
        min_connections = values.data.get("min_connections", 1)
        if v < min_connections:
            raise ValueError("max_connections must be >= min_connections")
        return v
