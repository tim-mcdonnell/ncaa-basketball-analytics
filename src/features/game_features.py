"""
Game features module.

This module contains functions for calculating game-level features.
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


def calculate_game_features(
    conn_id: str,
    database: str,
    lookback_days: int = 30,
    execution_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate game features and store them in the database.

    This function:
    1. Extracts game data from the database
    2. Retrieves team and player features
    3. Calculates features for each game
    4. Stores the features in the game_features table

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

    # Create game_features table if it doesn't exist
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS game_features (
        game_id VARCHAR,
        feature_date DATE,
        feature_name VARCHAR,
        feature_value DOUBLE,
        lookback_days INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (game_id, feature_date, feature_name, lookback_days)
    )
    """
    hook.run_query(create_table_sql)

    # Get games data for the lookback period
    games_df = hook.get_polars_df("SELECT * FROM games WHERE date >= ?", {"1": lookback_date})

    # Get team features
    team_features_df = hook.get_polars_df(
        "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?",
        {"1": execution_date, "2": lookback_days},
    )

    # Get player features - removed as it's not currently used in feature calculation
    # Will be implemented in future versions when player-based game features are added

    # Check if we have data
    if games_df.is_empty():
        logger.warning(f"No games found for the period from {lookback_date} to {execution_date}")
        return {"games_processed": 0, "features_calculated": 0, "success": True}

    # Feature calculation results
    features_calculated = 0
    games_processed = 0

    # Process each game
    for game_row in games_df.iter_rows(named=True):
        game_id = game_row["game_id"]
        home_team = game_row["home_team"]
        away_team = game_row["away_team"]
        games_processed += 1

        # Get team features for home team
        home_team_features = {}
        if not team_features_df.is_empty():
            home_team_df = team_features_df.filter(pl.col("team_id") == home_team)
            for row in home_team_df.iter_rows(named=True):
                home_team_features[row["feature_name"]] = row["feature_value"]

        # Get team features for away team
        away_team_features = {}
        if not team_features_df.is_empty():
            away_team_df = team_features_df.filter(pl.col("team_id") == away_team)
            for row in away_team_df.iter_rows(named=True):
                away_team_features[row["feature_name"]] = row["feature_value"]

        # Generate game features

        # 1. Win percentage difference
        if "win_percentage" in home_team_features and "win_percentage" in away_team_features:
            win_pct_diff = (
                home_team_features["win_percentage"] - away_team_features["win_percentage"]
            )
            store_feature(
                hook, game_id, execution_date, "win_percentage_diff", win_pct_diff, lookback_days
            )
            features_calculated += 1

        # 2. Points per game difference
        if "points_per_game" in home_team_features and "points_per_game" in away_team_features:
            ppg_diff = home_team_features["points_per_game"] - away_team_features["points_per_game"]
            store_feature(
                hook, game_id, execution_date, "points_per_game_diff", ppg_diff, lookback_days
            )
            features_calculated += 1

        # 3. Home team advantage (historical win rate at home)
        # This would require additional data about historical home wins, we'll add a placeholder
        home_advantage = 0.6  # Placeholder value based on common home advantage in NCAA basketball
        store_feature(
            hook, game_id, execution_date, "home_advantage", home_advantage, lookback_days
        )
        features_calculated += 1

        # 4. Score prediction based on team points per game
        if "points_per_game" in home_team_features and "points_per_game" in away_team_features:
            predicted_home_score = home_team_features["points_per_game"]
            predicted_away_score = away_team_features["points_per_game"]
            store_feature(
                hook,
                game_id,
                execution_date,
                "predicted_home_score",
                predicted_home_score,
                lookback_days,
            )
            store_feature(
                hook,
                game_id,
                execution_date,
                "predicted_away_score",
                predicted_away_score,
                lookback_days,
            )
            features_calculated += 2

        # 5. Predicted point differential
        if "points_per_game" in home_team_features and "points_per_game" in away_team_features:
            predicted_point_diff = (
                home_team_features["points_per_game"] - away_team_features["points_per_game"]
            )
            # Apply home court adjustment
            adjusted_point_diff = (
                predicted_point_diff + 3.5
            )  # Common home court advantage in points
            store_feature(
                hook,
                game_id,
                execution_date,
                "predicted_point_diff",
                adjusted_point_diff,
                lookback_days,
            )
            features_calculated += 1

        # 6. Win probability for home team
        # Using a simple logistic function based on adjusted point differential
        if "points_per_game" in home_team_features and "points_per_game" in away_team_features:
            adjusted_point_diff = (
                home_team_features["points_per_game"] - away_team_features["points_per_game"] + 3.5
            )
            # Simple sigmoid function to convert point diff to probability
            import math

            win_probability = 1 / (1 + math.exp(-0.2 * adjusted_point_diff))
            store_feature(
                hook,
                game_id,
                execution_date,
                "home_win_probability",
                win_probability,
                lookback_days,
            )
            features_calculated += 1

    logger.info(f"Processed {games_processed} games, calculated {features_calculated} features")

    return {
        "games_processed": games_processed,
        "features_calculated": features_calculated,
        "execution_date": execution_date,
        "lookback_date": lookback_date,
        "success": True,
    }


def store_feature(
    hook: DuckDBHook,
    game_id: str,
    feature_date: str,
    feature_name: str,
    feature_value: float,
    lookback_days: int,
) -> None:
    """
    Store a game feature in the database.

    Args:
        hook: DuckDB hook to use for database operations
        game_id: ID of the game
        feature_date: Date for the feature (usually execution date)
        feature_name: Name of the feature
        feature_value: Value of the feature
        lookback_days: Number of days used for calculation
    """
    insert_sql = """
    INSERT OR REPLACE INTO game_features
    (game_id, feature_date, feature_name, feature_value, lookback_days)
    VALUES (?, ?, ?, ?, ?)
    """
    hook.run_query(
        insert_sql,
        {
            "1": game_id,
            "2": feature_date,
            "3": feature_name,
            "4": feature_value,
            "5": lookback_days,
        },
    )
