from unittest.mock import patch, MagicMock

from airflow.hooks.duckdb_hook import DuckDBHook


class TestDuckDBHook:
    """Tests for the DuckDB hook implementation."""

    def test_hook_initialization(self):
        """Test that the hook initializes with the correct parameters."""
        # Arrange
        conn_id = "duckdb_default"
        database = "test_db.duckdb"

        # Act
        hook = DuckDBHook(conn_id=conn_id, database=database)

        # Assert
        assert hook.conn_id == conn_id
        assert hook.database == database

    def test_get_conn(self):
        """Test that get_conn returns a valid connection."""
        # Arrange
        hook = DuckDBHook(database=":memory:")

        # Act
        conn = hook.get_conn()

        # Assert
        assert conn is not None
        # Verify we can execute a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        assert result[0] == 1

    def test_get_records(self):
        """Test that get_records returns query results."""
        # Arrange
        hook = DuckDBHook(database=":memory:")

        # Act
        records = hook.get_records("SELECT 1 as test, 2 as test2")

        # Assert
        assert len(records) == 1
        assert records[0][0] == 1
        assert records[0][1] == 2

    def test_get_pandas_df(self):
        """Test that get_pandas_df returns a pandas DataFrame."""
        # Arrange
        hook = DuckDBHook(database=":memory:")

        # Act
        df = hook.get_pandas_df("SELECT 1 as test, 2 as test2")

        # Assert
        assert df is not None
        assert df.shape == (1, 2)
        assert df.iloc[0]["test"] == 1
        assert df.iloc[0]["test2"] == 2

    def test_get_polars_df(self):
        """Test that get_polars_df returns a polars DataFrame."""
        # Arrange
        hook = DuckDBHook(database=":memory:")

        # Act
        df = hook.get_polars_df("SELECT 1 as test, 2 as test2")

        # Assert
        assert df is not None
        assert df.shape == (1, 2)
        assert df.select("test").row(0)[0] == 1
        assert df.select("test2").row(0)[0] == 2

    def test_run_query(self):
        """Test that run_query executes a query successfully."""
        # Arrange
        hook = DuckDBHook(database=":memory:")

        # Act & Assert (no exception should be raised)
        hook.run_query("CREATE TABLE test (id INTEGER, name VARCHAR)")
        hook.run_query("INSERT INTO test VALUES (1, 'test')")

        # Verify data was inserted
        result = hook.get_records("SELECT * FROM test")
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == "test"

    @patch("airflow.hooks.base.BaseHook.get_connection")
    def test_connection_from_airflow_conn(self, mock_get_connection):
        """Test that the hook can use connection info from Airflow connections."""
        # Arrange
        mock_conn = MagicMock()
        mock_conn.host = "/path/to/db.duckdb"
        mock_get_connection.return_value = mock_conn

        # Act
        hook = DuckDBHook(conn_id="duckdb_test")

        # Assert
        mock_get_connection.assert_called_once_with("duckdb_test")
        assert hook.database == "/path/to/db.duckdb"
