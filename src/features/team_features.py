"""
Team features module.

This module contains functions for calculating team-level features.
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


from src.features.teams.basic_stats import (
    WinsFeature,
    LossesFeature,
    WinPercentageFeature,
    PointsPerGameFeature,
)


def calculate_team_features(
    conn_id: str,
    database: str,
    lookback_days: int = 30,
    execution_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate team features and store them in the database.

    This function:
    1. Extracts team and game data from the database
    2. Calculates features for each team
    3. Stores the features in the team_features table

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

    # Create team_features table if it doesn't exist
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS team_features (
        team_id VARCHAR,
        feature_date DATE,
        feature_name VARCHAR,
        feature_value DOUBLE,
        lookback_days INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (team_id, feature_date, feature_name, lookback_days)
    )
    """
    hook.run_query(create_table_sql)

    # Get teams data
    teams_df = hook.get_polars_df("SELECT * FROM teams")

    # Get games data for the lookback period
    games_df = hook.get_polars_df("SELECT * FROM games WHERE date >= ?", {"1": lookback_date})

    # Check if we have data
    if teams_df.is_empty():
        logger.warning("No teams found in the database")
        return {"teams_processed": 0, "features_calculated": 0, "success": False}

    if games_df.is_empty():
        logger.warning(f"No games found for the period from {lookback_date} to {execution_date}")
        return {"teams_processed": 0, "features_calculated": 0, "success": True}

    # Initialize feature calculators
    wins_feature = WinsFeature()
    losses_feature = LossesFeature()
    win_pct_feature = WinPercentageFeature()
    ppg_feature = PointsPerGameFeature()

    # Feature calculation results
    features_calculated = 0
    teams_processed = 0

    # Process each team
    for team_row in teams_df.iter_rows(named=True):
        team_id = team_row["team_id"]
        teams_processed += 1

        # Prepare data for home games
        home_games = games_df.filter(pl.col("home_team") == team_id)
        home_games = home_games.with_columns(
            pl.lit(team_id).alias("team_id"),
            pl.col("home_score").alias("team_score"),
            pl.col("away_score").alias("opponent_score"),
            pl.col("away_team").alias("opponent_id"),
        )

        # Prepare data for away games
        away_games = games_df.filter(pl.col("away_team") == team_id)
        away_games = away_games.with_columns(
            pl.lit(team_id).alias("team_id"),
            pl.col("away_score").alias("team_score"),
            pl.col("home_score").alias("opponent_score"),
            pl.col("home_team").alias("opponent_id"),
        )

        # Combine home and away games
        team_games = pl.concat(
            [
                home_games.select(
                    ["game_id", "team_id", "team_score", "opponent_score", "opponent_id", "date"]
                ),
                away_games.select(
                    ["game_id", "team_id", "team_score", "opponent_score", "opponent_id", "date"]
                ),
            ]
        )

        # Skip if no games for this team
        if team_games.is_empty():
            logger.info(f"No games found for team {team_id} in the lookback period")
            continue

        # Prepare data for feature calculation
        data = {"team_id": team_id, "games": team_games}

        # Calculate features
        feature_values = {
            "wins": wins_feature.compute(data),
            "losses": losses_feature.compute(data),
            "win_percentage": win_pct_feature.compute(
                {**data, "wins": wins_feature.compute(data), "losses": losses_feature.compute(data)}
            ),
            "points_per_game": ppg_feature.compute(data),
        }

        # Store features in the database
        for feature_name, feature_value in feature_values.items():
            insert_sql = """
            INSERT OR REPLACE INTO team_features
            (team_id, feature_date, feature_name, feature_value, lookback_days)
            VALUES (?, ?, ?, ?, ?)
            """
            hook.run_query(
                insert_sql,
                {
                    "1": team_id,
                    "2": execution_date,
                    "3": feature_name,
                    "4": feature_value,
                    "5": lookback_days,
                },
            )
            features_calculated += 1

    logger.info(f"Processed {teams_processed} teams, calculated {features_calculated} features")

    return {
        "teams_processed": teams_processed,
        "features_calculated": features_calculated,
        "execution_date": execution_date,
        "lookback_date": lookback_date,
        "success": True,
    }
