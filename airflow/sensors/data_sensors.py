"""
Data sensors for Apache Airflow.
These sensors detect data availability in DuckDB tables.
"""

from typing import Any, Dict, Optional

from airflow.sensors.base import BaseSensorOperator
from airflow.hooks.duckdb_hook import DuckDBHook
from airflow.utils.decorators import apply_defaults


class DuckDBTableSensor(BaseSensorOperator):
    """
    Sensor that checks if a DuckDB table exists and optionally has enough rows.

    :param conn_id: Connection ID to retrieve connection info from Airflow connections.
    :param database: Path to the DuckDB database file.
    :param table: Name of the table to check for existence.
    :param min_rows: Minimum number of rows the table should have (optional).
    """

    @apply_defaults
    def __init__(
        self, conn_id: str, database: str, table: str, min_rows: Optional[int] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.database = database
        self.table = table
        self.min_rows = min_rows

    def poke(self, context: Dict[str, Any]) -> bool:
        """
        Check if the specified table exists and has enough rows.

        :param context: Airflow context dictionary
        :return: True if the table exists and meets criteria, False otherwise
        """
        hook = DuckDBHook(conn_id=self.conn_id, database=self.database)

        # Check if table exists
        exists_query = f"""
        SELECT 1
        FROM information_schema.tables
        WHERE table_name = '{self.table}'
        """
        results = hook.get_records(exists_query)

        if not results:
            self.log.info(f"Table {self.table} does not exist.")
            return False

        # If min_rows is specified, check row count
        if self.min_rows is not None:
            count_query = f"SELECT COUNT(*) FROM {self.table}"
            count_result = hook.get_records(count_query)
            row_count = int(count_result[0][0])

            if row_count < self.min_rows:
                self.log.info(
                    f"Table {self.table} has {row_count} rows, less than minimum {self.min_rows}."
                )
                return False

            self.log.info(
                f"Table {self.table} has {row_count} rows, which meets minimum of {self.min_rows}."
            )
        else:
            self.log.info(f"Table {self.table} exists.")

        return True


class NewDataSensor(BaseSensorOperator):
    """
    Sensor that checks if new data is available in a DuckDB table since a timestamp.

    :param conn_id: Connection ID to retrieve connection info from Airflow connections.
    :param database: Path to the DuckDB database file.
    :param table: Name of the table to check for new data (not required if sql is provided).
    :param date_column: Name of the column containing the date/timestamp (not required if sql is provided).
    :param execution_date: Template reference to execution_date (e.g. "{{ execution_date }}").
    :param min_count: Minimum number of new records required (default: 1).
    :param sql: Custom SQL to check for new data. Will override table and date_column if provided.
    """

    template_fields = ("execution_date", "sql")

    @apply_defaults
    def __init__(
        self,
        conn_id: str,
        database: str,
        execution_date: str,
        table: Optional[str] = None,
        date_column: Optional[str] = None,
        min_count: int = 1,
        sql: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.database = database
        self.table = table
        self.date_column = date_column
        self.execution_date = execution_date
        self.min_count = min_count
        self.sql = sql

    def poke(self, context: Dict[str, Any]) -> bool:
        """
        Check if new data is available since the last execution date.

        :param context: Airflow context dictionary
        :return: True if new data is available, False otherwise
        """
        hook = DuckDBHook(conn_id=self.conn_id, database=self.database)

        # Determine which query to use
        if self.sql:
            query = self.sql
        else:
            if not self.table or not self.date_column:
                raise ValueError("Either provide a custom SQL query or both table and date_column.")

            query = f"""
            SELECT COUNT(*)
            FROM {self.table}
            WHERE {self.date_column} > '{self.execution_date}'
            """

        # Execute query
        results = hook.get_records(query)
        count = int(results[0][0])

        if count >= self.min_count:
            self.log.info(
                f"Found {count} new records, which meets the minimum of {self.min_count}."
            )
            return True
        else:
            self.log.info(f"Found only {count} new records, less than minimum of {self.min_count}.")
            return False
