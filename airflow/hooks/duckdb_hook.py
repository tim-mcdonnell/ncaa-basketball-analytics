"""
DuckDB Hook for Apache Airflow.
This hook provides connectivity to DuckDB databases.
"""

from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import polars as pl
from airflow.hooks.base import BaseHook


class DuckDBHook(BaseHook):
    """
    Hook for DuckDB.

    Interact with DuckDB databases using this hook. It provides methods to execute
    queries and retrieve results in different formats.

    :param conn_id: Connection ID to retrieve connection info from Airflow connections.
    :param database: Path to the DuckDB database file (can be ":memory:" for in-memory DB).
                    If conn_id is provided, this will be overridden by the connection info.
    :param read_only: Whether to open the database in read-only mode.
    """

    conn_type = "duckdb"

    def __init__(
        self,
        conn_id: Optional[str] = None,
        database: Optional[str] = None,
        read_only: bool = False,
    ):
        super().__init__()
        self.conn_id = conn_id
        self.database = database
        self.read_only = read_only

        if conn_id:
            conn = self.get_connection(conn_id)
            if conn.host and not database:
                self.database = conn.host  # DuckDB filename stored in host field

    def get_conn(self) -> duckdb.DuckDBPyConnection:
        """
        Get a DuckDB connection.

        :return: DuckDB connection object
        """
        conn = duckdb.connect(database=self.database, read_only=self.read_only)
        return conn

    def run_query(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute the SQL query with optional parameters.

        :param sql: SQL query to execute
        :param parameters: Parameters to bind to the query (optional)
        """
        conn = self.get_conn()
        try:
            if parameters:
                conn.execute(sql, parameters)
            else:
                conn.execute(sql)
        finally:
            conn.close()

    def get_records(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """
        Execute the SQL query and return the results as a list of tuples.

        :param sql: SQL query to execute
        :param parameters: Parameters to bind to the query (optional)
        :return: List of tuples containing the query results
        """
        conn = self.get_conn()
        try:
            if parameters:
                result = conn.execute(sql, parameters).fetchall()
            else:
                result = conn.execute(sql).fetchall()
            return result
        finally:
            conn.close()

    def get_pandas_df(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute the SQL query and return the results as a pandas DataFrame.

        :param sql: SQL query to execute
        :param parameters: Parameters to bind to the query (optional)
        :return: Pandas DataFrame containing the query results
        """
        conn = self.get_conn()
        try:
            if parameters:
                df = conn.execute(sql, parameters).df()
            else:
                df = conn.execute(sql).df()
            return df
        finally:
            conn.close()

    def get_polars_df(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
        """
        Execute the SQL query and return the results as a polars DataFrame.

        :param sql: SQL query to execute
        :param parameters: Parameters to bind to the query (optional)
        :return: Polars DataFrame containing the query results
        """
        # Get pandas dataframe first
        pandas_df = self.get_pandas_df(sql, parameters)
        # Convert to polars
        return pl.from_pandas(pandas_df)

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        :param table_name: Name of the table to check
        :return: True if the table exists, False otherwise
        """
        sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        result = self.get_records(sql)
        return len(result) > 0

    def get_table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.

        :param table_name: Name of the table to check
        :return: Number of rows in the table
        """
        sql = f"SELECT COUNT(*) FROM {table_name};"
        result = self.get_records(sql)
        return int(result[0][0]) if result else 0
