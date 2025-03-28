"""
Module for fetching upcoming games.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import DuckDBHook with try/except to handle potential absence of Airflow
try:
    from airflow.hooks.duckdb_hook import DuckDBHook
except ImportError:
    # Define a mock for when running outside Airflow
    class DuckDBHook:
        """Mock DuckDBHook for when Airflow is not available."""

        def __init__(self, **kwargs):
            pass


# Import ESPN API client
from src.data.api.espn_client import ESPNApiClient


def fetch_upcoming_games(
    conn_id: str,
    database: str,
    days_ahead: int = 7,
    execution_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch upcoming games from the ESPN API and store in the database.

    Args:
        conn_id: Connection ID for DuckDB connection
        database: Path to the DuckDB database file
        days_ahead: Number of days ahead to fetch games for
        execution_date: Execution date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dictionary with fetch results
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Set execution date to today if not provided
    if execution_date is None:
        execution_date = datetime.now().strftime("%Y-%m-%d")

    # Calculate end date
    end_date = (
        datetime.strptime(execution_date, "%Y-%m-%d") + timedelta(days=days_ahead)
    ).strftime("%Y-%m-%d")

    try:
        # Initialize hook and API client
        hook = DuckDBHook(conn_id=conn_id, database=database)
        api_client = ESPNApiClient()

        # Ensure games table exists
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS games (
            game_id VARCHAR PRIMARY KEY,
            home_team VARCHAR NOT NULL,
            away_team VARCHAR NOT NULL,
            date DATE NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            status VARCHAR,
            venue VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        hook.run_query(create_table_sql)

        # Get existing game IDs to avoid duplicates
        existing_game_ids = set()
        existing_games_sql = "SELECT game_id, date FROM games"
        for record in hook.get_records(existing_games_sql):
            existing_game_ids.add(record[0])

        # Fetch upcoming games from ESPN API
        logger.info(f"Fetching upcoming games from {execution_date} to {end_date}")
        games = api_client.get_games(start_date=execution_date, end_date=end_date)
        logger.info(f"Retrieved {len(games)} games")

        # Track new games
        new_games = 0

        # Store games in database, skipping existing ones
        for game in games:
            game_id = game.get("game_id")

            # Skip if game already exists
            if game_id in existing_game_ids:
                continue

            # Insert new game
            insert_sql = """
            INSERT INTO games (
                game_id, home_team, away_team, date, status, venue
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """
            params = (
                game_id,
                game.get("home_team"),
                game.get("away_team"),
                game.get("date"),
                game.get("status", "scheduled"),
                game.get("venue"),
            )
            hook.run_query(insert_sql, dict(zip(range(1, len(params) + 1), params)))
            new_games += 1

        logger.info(f"Added {new_games} new games to the database")

        return {
            "success": True,
            "games_fetched": len(games),
            "new_games": new_games,
            "execution_date": execution_date,
            "end_date": end_date,
        }

    except Exception as e:
        logger.error(f"Error fetching upcoming games: {str(e)}")
        return {"success": False, "error": str(e), "execution_date": execution_date}
