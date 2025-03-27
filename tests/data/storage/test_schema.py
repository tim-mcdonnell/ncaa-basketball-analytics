import tempfile
import os

import pytest
import duckdb

from src.data.storage.db import DatabaseManager
from src.data.storage.schema import get_schema_definitions


def test_schema_structure():
    """Test that the schema structure matches the expected design."""
    # Get the schema definitions
    schema_definitions = get_schema_definitions()

    # Verify we have definitions for all expected tables
    expected_tables = [
        "raw_teams",
        "raw_games",
        "raw_players",
        "dim_teams",
        "dim_players",
        "dim_seasons",
        "fact_games",
        "fact_player_stats",
    ]

    for table in expected_tables:
        assert table in schema_definitions, f"Missing schema definition for {table}"


def test_schema_constraints():
    """Test that the schema enforces data integrity constraints."""
    # Use a temporary directory instead of a file for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = os.path.join(temp_dir, "test.duckdb")

        try:
            # Initialize the database with schema
            db_manager = DatabaseManager(db_path=temp_db_path)
            db_manager.initialize_schema()

            connection = db_manager.get_connection()

            # Test primary key constraint in raw_teams
            with pytest.raises(duckdb.ConstraintException):
                # Try to insert a row with duplicate primary key
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO raw_teams (id, team_id, raw_data, source_url, processing_version)
                    VALUES (uuid_generate_v4(), 'MICH', '{"name": "Michigan"}', 'http://example.com', '1.0')
                """)

                # Try to insert another row with the same team_id (should violate unique constraint)
                cursor.execute("""
                    INSERT INTO raw_teams (id, team_id, raw_data, source_url, processing_version)
                    VALUES (uuid_generate_v4(), 'MICH', '{"name": "Michigan"}', 'http://example.com', '1.0')
                """)

            # Test not null constraints in dim_teams
            with pytest.raises(duckdb.ConstraintException):
                cursor = connection.cursor()
                # Create dim_teams table if needed
                cursor.execute("""
                    INSERT INTO dim_teams (team_id, name, conference, division)
                    VALUES (NULL, 'Michigan', 'Big Ten', 'East')
                """)

            connection.close()

        finally:
            # Cleanup is handled by the context manager
            pass


def test_referential_integrity():
    """Test that referential integrity is enforced between related tables."""
    # Use a temporary directory instead of a file for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = os.path.join(temp_dir, "test.duckdb")

        try:
            # Initialize the database with schema
            db_manager = DatabaseManager(db_path=temp_db_path)
            db_manager.initialize_schema()

            connection = db_manager.get_connection()
            cursor = connection.cursor()

            # Set up test data
            cursor.execute("""
                INSERT INTO dim_teams (team_id, name, conference, division)
                VALUES ('MICH', 'Michigan', 'Big Ten', 'East')
            """)

            cursor.execute("""
                INSERT INTO dim_seasons (season_id, year, type)
                VALUES (1, 2023, 'Regular')
            """)

            # This should work - valid foreign keys
            cursor.execute("""
                INSERT INTO fact_games (
                    game_id, season_id, home_team_id, away_team_id,
                    home_score, away_score, game_date, venue, status
                )
                VALUES (
                    'GAME001', 1, 'MICH', 'MICH',
                    75, 65, '2023-11-01', 'Michigan Stadium', 'FINAL'
                )
            """)

            # This should fail - invalid team_id foreign key
            with pytest.raises(duckdb.ConstraintException):
                cursor.execute("""
                    INSERT INTO fact_games (
                        game_id, season_id, home_team_id, away_team_id,
                        home_score, away_score, game_date, venue, status
                    )
                    VALUES (
                        'GAME002', 1, 'INVALID', 'MICH',
                        75, 65, '2023-11-01', 'Michigan Stadium', 'FINAL'
                    )
                """)

            connection.close()

        finally:
            # Cleanup is handled by the context manager
            pass
