"""
Database connection and management module for DuckDB.
"""

import uuid
from pathlib import Path
from typing import Optional

import duckdb
from pydantic import ValidationError

from src.data.storage.schema import get_schema_definitions
from src.config.models import DatabaseConfig
from src.config.loaders import load_config


class DatabaseManager:
    """
    Manages the DuckDB database connection and schema initialization.

    This class provides methods to connect to DuckDB, initialize database schema,
    and manage database operations.

    Attributes:
        db_path: Path to the DuckDB database file
        connection: Active DuckDB connection
    """

    _DEFAULT_CONFIG_PATH = "config/db_config.yaml"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.

        Args:
            db_path: Optional path to the database file.
                    If None, uses the default path from configuration.
        """
        # Ensure db_path is absolute or relative to project root
        if db_path:
            self.db_path = str(Path(db_path).resolve())
        else:
            self.db_path = str(self._get_default_db_path().resolve())

        self.connection: Optional[duckdb.DuckDBPyConnection] = None

    def _get_default_db_path(self) -> Path:
        """
        Get the default database path from configuration.

        Returns:
            Path object pointing to the default database location relative to project root.
        """
        # Load path from config file
        try:
            config = load_config(self._DEFAULT_CONFIG_PATH, DatabaseConfig)
            # Assume path is relative to project root
            project_root = Path(__file__).parents[3]
            return project_root / config.path
        except (FileNotFoundError, ValidationError) as e:
            # Provide a more informative error message if config loading fails
            raise RuntimeError(
                f"Failed to load database configuration from {self._DEFAULT_CONFIG_PATH}: {e}"
            ) from e

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get a connection to the DuckDB database.

        Returns:
            DuckDB connection object
        """
        if self.connection is None or self.connection.is_closed():
            # Ensure parent directory exists before connecting
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            # Connect to the database
            self.connection = duckdb.connect(self.db_path)

            # Register a UUID generation function
            self._register_uuid_function()

        return self.connection

    def _register_uuid_function(self):
        """Register a UUID generation function in DuckDB."""

        # Define a Python function to generate UUIDs
        def generate_uuid() -> str:
            return str(uuid.uuid4())

        # Register the function in DuckDB
        conn = self.get_connection()
        conn.create_function("uuid_generate_v4", generate_uuid, [], duckdb.typing.VARCHAR)

    def close_connection(self):
        """Close the database connection if it's open."""
        if self.connection is not None and not self.connection.is_closed():
            self.connection.close()
        self.connection = None

    def initialize_schema(self):
        """
        Initialize the database schema with all required tables.

        This method creates all tables defined in the schema definitions
        if they don't already exist.
        """
        connection = self.get_connection()
        schema_definitions = get_schema_definitions()

        for _table_name, table_sql in schema_definitions.items():
            connection.execute(table_sql)

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.close_connection()
