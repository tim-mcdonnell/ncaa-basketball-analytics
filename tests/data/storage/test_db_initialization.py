import os
import tempfile


from src.data.storage.db import DatabaseManager


def test_database_initialization():
    """Test that the database file is created and initialized properly."""
    # Use a temporary directory instead of a file for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = os.path.join(temp_dir, "test.duckdb")

        try:
            # Initialize the database
            db_manager = DatabaseManager(db_path=temp_db_path)

            # Check that we can connect to the database
            connection = db_manager.get_connection()
            assert connection is not None

            # Check that we can execute a simple query
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

            # Close the connection
            connection.close()

        finally:
            # Cleanup is handled by the context manager
            pass


def test_schema_creation():
    """Test that the schema is created properly with all required tables."""
    # Use a temporary directory instead of a file for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = os.path.join(temp_dir, "test.duckdb")

        try:
            # Initialize the database with schema
            db_manager = DatabaseManager(db_path=temp_db_path)
            db_manager.initialize_schema()

            # Check that all required tables exist
            connection = db_manager.get_connection()
            cursor = connection.cursor()

            # Check raw layer tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'raw_%'"
            )
            raw_tables = {row[0] for row in cursor.fetchall()}
            assert "raw_teams" in raw_tables
            assert "raw_games" in raw_tables
            assert "raw_players" in raw_tables

            # Check dimension tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'dim_%'"
            )
            dim_tables = {row[0] for row in cursor.fetchall()}
            assert "dim_teams" in dim_tables
            assert "dim_players" in dim_tables
            assert "dim_seasons" in dim_tables

            # Check fact tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fact_%'"
            )
            fact_tables = {row[0] for row in cursor.fetchall()}
            assert "fact_games" in fact_tables
            assert "fact_player_stats" in fact_tables

            # Close the connection
            connection.close()

        finally:
            # Cleanup is handled by the context manager
            pass
