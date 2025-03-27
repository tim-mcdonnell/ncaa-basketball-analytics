"""
Database connection and management module for DuckDB.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

import duckdb

from src.data.storage.schema import get_schema_definitions


class DatabaseManager:
    """
    Manages the DuckDB database connection and schema initialization.

    This class provides methods to connect to DuckDB, initialize database schema,
    and manage database operations.

    Attributes:
        db_path: Path to the DuckDB database file
        connection: Active DuckDB connection
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.

        Args:
            db_path: Optional path to the database file.
                    If None, uses the default path from configuration.
        """
        self.db_path = db_path or str(self._get_default_db_path())
        self.connection = None

    def _get_default_db_path(self) -> Path:
        """
        Get the default database path from configuration.

        Returns:
            Path object pointing to the default database location
        """
        # TODO: Read this from configuration file
        base_dir = Path(__file__).parents[3]  # Project root
        data_dir = base_dir / "data"

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        return data_dir / "ncaa_basketball.duckdb"

    def get_connection(self):
        """
        Get a connection to the DuckDB database.

        Returns:
            DuckDB connection object
        """
        if self.connection is None:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Connect to the database
            self.connection = duckdb.connect(self.db_path)

            # Register a UUID generation function
            self._register_uuid_function()

        return self.connection

    def _register_uuid_function(self):
        """Register a UUID generation function in DuckDB."""

        # Define a Python function to generate UUIDs
        def generate_uuid():
            return str(uuid.uuid4())

        # Register the function in DuckDB
        self.connection.create_function(
            "uuid_generate_v4", generate_uuid, [], duckdb.typing.VARCHAR
        )

    def close_connection(self):
        """Close the database connection if it's open."""
        if self.connection is not None:
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

        for table_name, table_sql in schema_definitions.items():
            connection.execute(table_sql)

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.close_connection()
