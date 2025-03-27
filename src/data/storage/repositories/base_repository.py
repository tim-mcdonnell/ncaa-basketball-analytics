"""
Base repository for data access.

This module defines the BaseRepository class that provides common functionality
for all repository implementations.
"""

from typing import Any, Dict, List, Optional, Union

import duckdb
import polars as pl

from src.data.storage.db import DatabaseManager


class BaseRepository:
    """
    Base repository class providing common database operations.

    This class serves as a foundation for all repositories, implementing common
    data access patterns and database operations.

    Attributes:
        db_manager: Database manager instance
        table_name: Name of the database table this repository manages
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None, table_name: str = None):
        """
        Initialize the repository.

        Args:
            db_manager: Optional database manager instance. If None, creates a new one.
            table_name: Name of the database table this repository manages
        """
        self.db_manager = db_manager or DatabaseManager()
        self.table_name = table_name

        if not self.table_name:
            raise ValueError("table_name must be provided")

    def execute_query(
        self, query: str, params: Optional[Union[List[Any], Dict[str, Any]]] = None
    ) -> duckdb.DuckDBPyConnection:
        """
        Execute a SQL query with optional parameters.

        Args:
            query: SQL query to execute
            params: Optional parameters for the query as a list or dict

        Returns:
            DuckDB cursor for the executed query
        """
        connection = self.db_manager.get_connection()

        if params is not None:
            if isinstance(params, dict):
                # Convert parameter placeholders
                # Replace named parameters (:param) with positional ones (?)
                # and build a list of parameter values in the right order
                param_values = []
                if "?" not in query and any(f":{key}" in query for key in params.keys()):
                    # Handle named parameters with :name format
                    for key, value in params.items():
                        query = query.replace(f":{key}", "?")
                        param_values.append(value)
                else:
                    # For queries with ? placeholders
                    param_values = list(params.values())
                return connection.execute(query, param_values)
            else:
                # List of parameters
                return connection.execute(query, params)
        return connection.execute(query)

    def query_to_polars(
        self, query: str, params: Optional[Union[List[Any], Dict[str, Any]]] = None
    ) -> pl.DataFrame:
        """
        Execute a query and return the results as a Polars DataFrame.

        Args:
            query: SQL query to execute
            params: Optional parameters for the query as a list or dict

        Returns:
            Polars DataFrame containing query results
        """
        # No need to get connection here since execute_query does it
        result = self.execute_query(query, params)
        return result.pl()

    def get_all(self) -> pl.DataFrame:
        """
        Get all records from the repository's table.

        Returns:
            Polars DataFrame containing all records
        """
        query = f"SELECT * FROM {self.table_name}"
        return self.query_to_polars(query)

    def get_by_id(self, id_value: Union[str, int], id_column: str = None) -> Optional[pl.DataFrame]:
        """
        Get a record by its ID.

        Args:
            id_value: ID value to look up
            id_column: Optional column name to use. If None, uses table_name + '_id'
                       or 'id' if table_name is None.

        Returns:
            Polars DataFrame with the matching record, or None if not found
        """
        if id_column is None:
            # Try to determine appropriate ID column
            if self.table_name and not self.table_name.startswith(("dim_", "fact_", "raw_")):
                id_column = f"{self.table_name}_id"
            else:
                # Handle prefix for dimension/fact tables
                parts = self.table_name.split("_", 1)
                if len(parts) > 1 and parts[0] in ("dim", "fact", "raw"):
                    if parts[1].endswith("s"):  # Handle plural table names
                        id_column = f"{parts[1][:-1]}_id"  # Remove 's' from end
                    else:
                        id_column = f"{parts[1]}_id"
                else:
                    id_column = "id"

        query = f"""
        SELECT * FROM {self.table_name}
        WHERE {id_column} = ?
        """

        result = self.query_to_polars(query, [id_value])

        return result if not result.is_empty() else None

    def insert(self, data: Dict[str, Any]) -> str:
        """
        Insert a record into the repository's table.

        Args:
            data: Dictionary of column names and values to insert

        Returns:
            ID of the inserted record
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data.keys()])

        query = f"""
        INSERT INTO {self.table_name} ({columns})
        VALUES ({placeholders})
        RETURNING *
        """

        result = self.execute_query(query, list(data.values()))
        return result.fetchone()[0]  # Return the ID

    def insert_many(self, data_frames: Union[pl.DataFrame, List[Dict[str, Any]]]) -> int:
        """
        Insert multiple records into the repository's table.

        Args:
            data_frames: Polars DataFrame or list of dictionaries with data to insert

        Returns:
            Number of inserted records
        """
        connection = self.db_manager.get_connection()

        if isinstance(data_frames, pl.DataFrame):
            # Insert directly from Polars DataFrame
            connection.execute(f"INSERT INTO {self.table_name} SELECT * FROM data_frames")
            return len(data_frames)

        if isinstance(data_frames, list) and all(isinstance(item, dict) for item in data_frames):
            # Insert from list of dictionaries
            if not data_frames:
                return 0

            # Get column names from first dictionary
            columns = data_frames[0].keys()
            placeholders = ", ".join(["?" for _ in columns])
            columns_str = ", ".join(columns)

            query = f"""
            INSERT INTO {self.table_name} ({columns_str})
            VALUES ({placeholders})
            """

            # Prepare values for batch insert
            values = [tuple(item.values()) for item in data_frames]

            # Execute batch insert
            cursor = connection.cursor()
            cursor.executemany(query, values)

            return len(data_frames)

        raise ValueError("data_frames must be a Polars DataFrame or a list of dictionaries")

    def update(self, id_value: Union[str, int], data: Dict[str, Any], id_column: str = None) -> int:
        """
        Update a record in the repository's table.

        Args:
            id_value: ID of the record to update
            data: Dictionary of column names and values to update
            id_column: Optional ID column name (defaults to table_name + '_id')

        Returns:
            Number of rows updated (0 or 1)
        """
        if id_column is None:
            # Use the same ID column determination as get_by_id
            if self.table_name and not self.table_name.startswith(("dim_", "fact_", "raw_")):
                id_column = f"{self.table_name}_id"
            else:
                # Handle prefix for dimension/fact tables
                parts = self.table_name.split("_", 1)
                if len(parts) > 1 and parts[0] in ("dim", "fact", "raw"):
                    if parts[1].endswith("s"):  # Handle plural table names
                        id_column = f"{parts[1][:-1]}_id"  # Remove 's' from end
                    else:
                        id_column = f"{parts[1]}_id"
                else:
                    id_column = "id"

        # Build SET clause
        set_clause = ", ".join([f"{key} = ?" for key in data.keys()])

        query = f"""
        UPDATE {self.table_name}
        SET {set_clause}
        WHERE {id_column} = ?
        """

        # Combine data values with ID value
        params = list(data.values())
        params.append(id_value)

        result = self.execute_query(query, params)
        return result.fetchone()[0]  # Return number of rows updated

    def delete(self, id_value: Union[str, int], id_column: str = None) -> int:
        """
        Delete a record from the repository's table.

        Args:
            id_value: ID of the record to delete
            id_column: Optional ID column name (defaults to table_name + '_id')

        Returns:
            Number of rows deleted (0 or 1)
        """
        if id_column is None:
            # Use the same ID column determination as get_by_id
            if self.table_name and not self.table_name.startswith(("dim_", "fact_", "raw_")):
                id_column = f"{self.table_name}_id"
            else:
                # Handle prefix for dimension/fact tables
                parts = self.table_name.split("_", 1)
                if len(parts) > 1 and parts[0] in ("dim", "fact", "raw"):
                    if parts[1].endswith("s"):  # Handle plural table names
                        id_column = f"{parts[1][:-1]}_id"  # Remove 's' from end
                    else:
                        id_column = f"{parts[1]}_id"
                else:
                    id_column = "id"

        query = f"""
        DELETE FROM {self.table_name}
        WHERE {id_column} = ?
        """

        result = self.execute_query(query, [id_value])
        return result.fetchone()[0]  # Return number of rows deleted
