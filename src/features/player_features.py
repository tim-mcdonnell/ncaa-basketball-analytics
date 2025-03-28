"""
Player features module.

This module contains functions for calculating player-level features.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import polars as pl
import logging

# Import DuckDBHook with try/except to handle potential absence of Airflow
try:
    from airflow.hooks.duckdb_hook import DuckDBHook
except ImportError:
    # Define a mock for when running outside Airflow
    class DuckDBHook:
        """Mock DuckDBHook for when Airflow is not available."""

        def __init__(self, **kwargs):
            pass


def calculate_player_features(
    conn_id: str,
    database: str,
    lookback_days: int = 30,
    execution_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate player features and store them in the database.

    This function:
    1. Extracts player and player stats data from the database
    2. Calculates features for each player
    3. Stores the features in the player_features table

    Args:
        conn_id: Connection ID for DuckDB connection
        database: Path to the DuckDB database file
        lookback_days: Number of days to look back for calculating features
        execution_date: Execution date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dictionary with summary statistics
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Set execution date to today if not provided
    if execution_date is None:
        execution_date = datetime.now().strftime("%Y-%m-%d")

    # Calculate the lookback date
    lookback_date = (
        datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    # Initialize hook
    hook = DuckDBHook(conn_id=conn_id, database=database)

    # Create player_features table if it doesn't exist
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS player_features (
        player_id VARCHAR,
        feature_date DATE,
        feature_name VARCHAR,
        feature_value DOUBLE,
        lookback_days INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (player_id, feature_date, feature_name, lookback_days)
    )
    """
    hook.run_query(create_table_sql)

    # Get players data
    players_df = hook.get_polars_df("SELECT * FROM players")

    # Get player stats data for the lookback period
    player_stats_df = hook.get_polars_df(
        "SELECT ps.* FROM player_stats ps JOIN games g ON ps.game_id = g.game_id WHERE g.date >= ?",
        {"1": lookback_date},
    )

    # Check if we have data
    if players_df.is_empty():
        logger.warning("No players found in the database")
        return {"players_processed": 0, "features_calculated": 0, "success": False}

    if player_stats_df.is_empty():
        logger.warning(
            f"No player stats found for the period from {lookback_date} to {execution_date}"
        )
        return {"players_processed": 0, "features_calculated": 0, "success": True}

    # Feature calculation results
    features_calculated = 0
    players_processed = 0

    # Process each player
    for player_row in players_df.iter_rows(named=True):
        player_id = player_row["player_id"]
        players_processed += 1

        # Get stats for this player
        player_stats = player_stats_df.filter(pl.col("player_id") == player_id)

        # Skip if no stats for this player
        if player_stats.is_empty():
            logger.info(f"No stats found for player {player_id} in the lookback period")
            continue

        # Calculate features for this player

        # 1. Games played
        games_played = player_stats.height
        store_feature(hook, player_id, execution_date, "games_played", games_played, lookback_days)
        features_calculated += 1

        # 2. Points per game
        ppg = player_stats["points"].mean()
        store_feature(hook, player_id, execution_date, "points_per_game", ppg, lookback_days)
        features_calculated += 1

        # 3. Rebounds per game
        rpg = player_stats["rebounds"].mean()
        store_feature(hook, player_id, execution_date, "rebounds_per_game", rpg, lookback_days)
        features_calculated += 1

        # 4. Assists per game
        apg = player_stats["assists"].mean()
        store_feature(hook, player_id, execution_date, "assists_per_game", apg, lookback_days)
        features_calculated += 1

        # 5. Steals per game
        spg = player_stats["steals"].mean()
        store_feature(hook, player_id, execution_date, "steals_per_game", spg, lookback_days)
        features_calculated += 1

        # 6. Blocks per game
        bpg = player_stats["blocks"].mean()
        store_feature(hook, player_id, execution_date, "blocks_per_game", bpg, lookback_days)
        features_calculated += 1

        # 7. Turnovers per game
        tpg = player_stats["turnovers"].mean()
        store_feature(hook, player_id, execution_date, "turnovers_per_game", tpg, lookback_days)
        features_calculated += 1

        # 8. Minutes per game
        mpg = player_stats["minutes"].mean()
        store_feature(hook, player_id, execution_date, "minutes_per_game", mpg, lookback_days)
        features_calculated += 1

        # 9. Points per minute
        points_per_minute = (
            (player_stats["points"].sum() / player_stats["minutes"].sum())
            if player_stats["minutes"].sum() > 0
            else 0
        )
        store_feature(
            hook, player_id, execution_date, "points_per_minute", points_per_minute, lookback_days
        )
        features_calculated += 1

        # 10. Efficiency rating (simple version: pts + reb + ast + stl + blk - to)
        efficiency = (
            player_stats["points"].mean()
            + player_stats["rebounds"].mean()
            + player_stats["assists"].mean()
            + player_stats["steals"].mean()
            + player_stats["blocks"].mean()
            - player_stats["turnovers"].mean()
        )
        store_feature(
            hook, player_id, execution_date, "efficiency_rating", efficiency, lookback_days
        )
        features_calculated += 1

    logger.info(f"Processed {players_processed} players, calculated {features_calculated} features")

    return {
        "players_processed": players_processed,
        "features_calculated": features_calculated,
        "execution_date": execution_date,
        "lookback_date": lookback_date,
        "success": True,
    }


def store_feature(
    hook: DuckDBHook,
    player_id: str,
    feature_date: str,
    feature_name: str,
    feature_value: float,
    lookback_days: int,
) -> None:
    """
    Store a player feature in the database.

    Args:
        hook: DuckDB hook to use for database operations
        player_id: ID of the player
        feature_date: Date for the feature (usually execution date)
        feature_name: Name of the feature
        feature_value: Value of the feature
        lookback_days: Number of days used for calculation
    """
    insert_sql = """
    INSERT OR REPLACE INTO player_features
    (player_id, feature_date, feature_name, feature_value, lookback_days)
    VALUES (?, ?, ?, ?, ?)
    """
    hook.run_query(
        insert_sql,
        {
            "1": player_id,
            "2": feature_date,
            "3": feature_name,
            "4": feature_value,
            "5": lookback_days,
        },
    )
