"""
Module for preparing prediction data.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import polars as pl

# Import DuckDBHook with try/except to handle potential absence of Airflow
try:
    from airflow.hooks.duckdb_hook import DuckDBHook
except ImportError:
    # Define a mock for when running outside Airflow
    class DuckDBHook:
        """Mock DuckDBHook for when Airflow is not available."""

        def __init__(self, **kwargs):
            pass


def prepare_prediction_data(
    conn_id: str,
    database: str,
    lookback_days: int = 30,
    execution_date: Optional[str] = None,
    output_path: str = "./data/predictions",
) -> Dict[str, Any]:
    """
    Prepare data for upcoming games that need predictions.

    This function:
    1. Retrieves upcoming scheduled games
    2. Gets team features for teams in these games
    3. Combines data to create prediction-ready dataset
    4. Saves dataset for use by prediction models

    Args:
        conn_id: Connection ID for DuckDB connection
        database: Path to the DuckDB database file
        lookback_days: Number of days to look back for team features
        execution_date: Execution date in YYYY-MM-DD format (defaults to today)
        output_path: Directory to save the output files

    Returns:
        Dictionary with preparation results
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Set execution date to today if not provided
    if execution_date is None:
        execution_date = datetime.now().strftime("%Y-%m-%d")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    try:
        # Initialize hook
        hook = DuckDBHook(conn_id=conn_id, database=database)

        # Get upcoming scheduled games
        logger.info(f"Fetching upcoming scheduled games from {execution_date}")
        games_df = hook.get_polars_df(
            "SELECT * FROM games WHERE status = 'scheduled' AND date >= ?", {"1": execution_date}
        )

        # Check if we have upcoming games
        if games_df.is_empty():
            logger.warning(f"No upcoming games found from {execution_date}")

            # Create empty prediction data file
            empty_df = pl.DataFrame({"game_id": [], "home_team": [], "away_team": [], "date": []})

            # Save empty data
            empty_df.write_parquet(os.path.join(output_path, "prediction_data.parquet"))

            # Save empty feature columns
            with open(os.path.join(output_path, "feature_columns.json"), "w") as f:
                json.dump([], f)

            return {
                "success": True,
                "games_processed": 0,
                "features_calculated": 0,
                "execution_date": execution_date,
            }

        # Get team features
        logger.info(f"Fetching team features with lookback days: {lookback_days}")
        team_features_df = hook.get_polars_df(
            "SELECT * FROM team_features WHERE feature_date = ? AND lookback_days = ?",
            {"1": execution_date, "2": lookback_days},
        )

        # Process data to create prediction dataset
        prediction_data = process_prediction_data(games_df, team_features_df)

        # Get feature columns (excluding identifiers)
        exclude_columns = ["game_id", "home_team", "away_team", "date"]
        feature_columns = [col for col in prediction_data.columns if col not in exclude_columns]

        # Save feature columns
        with open(os.path.join(output_path, "feature_columns.json"), "w") as f:
            json.dump(feature_columns, f)

        # Save prediction data
        prediction_data.write_parquet(os.path.join(output_path, "prediction_data.parquet"))

        logger.info(f"Processed {len(prediction_data)} games for prediction")

        return {
            "success": True,
            "games_processed": len(prediction_data),
            "features_calculated": len(feature_columns),
            "execution_date": execution_date,
        }

    except Exception as e:
        logger.error(f"Error preparing prediction data: {str(e)}")
        return {"success": False, "error": str(e), "execution_date": execution_date}


def process_prediction_data(games_df: pl.DataFrame, team_features_df: pl.DataFrame) -> pl.DataFrame:
    """
    Process games and team features into a prediction-ready dataset.

    Args:
        games_df: DataFrame with upcoming games
        team_features_df: DataFrame with team features

    Returns:
        DataFrame ready for prediction
    """
    # Create base DataFrame with game info
    prediction_data = games_df.select(["game_id", "home_team", "away_team", "date"])

    # If we have team features, add them to the prediction data
    if not team_features_df.is_empty():
        # Create dictionaries to map team ID to features
        team_features = {}

        # Process team features into a dictionary for easier lookup
        for team_id in team_features_df["team_id"].unique():
            team_data = team_features_df.filter(pl.col("team_id") == team_id)
            team_features[team_id] = {}

            for row in team_data.iter_rows(named=True):
                feature_name = row["feature_name"]
                feature_value = row["feature_value"]
                team_features[team_id][feature_name] = feature_value

        # Add columns for each feature
        unique_features = set()
        for features in team_features.values():
            unique_features.update(features.keys())

        # For each pair of teams, calculate feature differences
        # Add columns for feature differences
        feature_diffs = []

        for feature in unique_features:
            # Collect values for home and away teams
            home_values = []
            away_values = []

            for row in prediction_data.iter_rows(named=True):
                home_team = row["home_team"]
                away_team = row["away_team"]

                # Get feature values, defaulting to 0 if not found
                home_value = team_features.get(home_team, {}).get(feature, 0)
                away_value = team_features.get(away_team, {}).get(feature, 0)

                home_values.append(home_value)
                away_values.append(away_value)

            # Add home and away columns
            prediction_data = prediction_data.with_columns(
                [
                    pl.Series(f"home_{feature}", home_values),
                    pl.Series(f"away_{feature}", away_values),
                ]
            )

            # Calculate and add difference column
            prediction_data = prediction_data.with_columns(
                [(pl.col(f"home_{feature}") - pl.col(f"away_{feature}")).alias(f"{feature}_diff")]
            )

            # Track feature difference column
            feature_diffs.append(f"{feature}_diff")

    return prediction_data
