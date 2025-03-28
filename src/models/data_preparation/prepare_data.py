"""
Module for preparing training data for model training.
"""

import os
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

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


def prepare_training_data(
    conn_id: str,
    database: str,
    lookback_days: int = 365,
    execution_date: Optional[str] = None,
    output_path: str = "./data/model_training",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Prepare training data for model training.

    This function:
    1. Extracts game data and features from the database
    2. Processes and combines features into a training dataset
    3. Splits data into training, validation, and test sets
    4. Saves these sets as parquet files

    Args:
        conn_id: Connection ID for DuckDB connection
        database: Path to the DuckDB database file
        lookback_days: Number of days to look back for data
        execution_date: Execution date in YYYY-MM-DD format (defaults to today)
        output_path: Directory to save the output files
        train_ratio: Ratio of data to use for training (default: 0.7)
        val_ratio: Ratio of data to use for validation (default: 0.15)
        test_ratio: Ratio of data to use for testing (default: 0.15)

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

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Initialize hook
    hook = DuckDBHook(conn_id=conn_id, database=database)

    # Get completed games data for the lookback period
    games_df = hook.get_polars_df(
        "SELECT * FROM games WHERE date <= ? AND date >= ? AND status = 'final'",
        {"1": execution_date, "2": lookback_date},
    )

    # Check if we have game data
    if games_df.is_empty():
        logger.warning(f"No games found for the period from {lookback_date} to {execution_date}")
        return {"games_processed": 0, "features_calculated": 0, "success": False}

    # Get game features
    game_features_df = hook.get_polars_df(
        "SELECT * FROM game_features WHERE feature_date <= ? AND lookback_days = ?",
        {"1": execution_date, "2": lookback_days},
    )

    # Get team features
    team_features_df = hook.get_polars_df(
        "SELECT * FROM team_features WHERE feature_date <= ? AND lookback_days = ?",
        {"1": execution_date, "2": lookback_days},
    )

    # Process and prepare the data
    processed_data = process_training_data(games_df, game_features_df, team_features_df)

    # If no data was processed, return
    if processed_data.is_empty():
        logger.warning("No data could be processed for training")
        return {"games_processed": 0, "features_calculated": 0, "success": False}

    # Get feature columns (excluding identifiers and target variables)
    exclude_columns = ["game_id", "home_team", "away_team", "date", "home_win", "point_diff"]
    feature_columns = [col for col in processed_data.columns if col not in exclude_columns]

    # Save feature columns for later use during inference
    with open(os.path.join(output_path, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)

    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = split_data(processed_data, train_ratio, val_ratio, test_ratio)

    # Save the datasets as parquet files
    train_data.write_parquet(os.path.join(output_path, "train_data.parquet"))
    val_data.write_parquet(os.path.join(output_path, "val_data.parquet"))
    test_data.write_parquet(os.path.join(output_path, "test_data.parquet"))

    logger.info(f"Processed {len(processed_data)} games for training")
    logger.info(
        f"Split data: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test"
    )

    return {
        "games_processed": len(processed_data),
        "features_calculated": len(feature_columns),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "execution_date": execution_date,
        "lookback_date": lookback_date,
        "success": True,
    }


def process_training_data(
    games_df: pl.DataFrame, game_features_df: pl.DataFrame, team_features_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Process and combine game data and features into a training dataset.

    Args:
        games_df: DataFrame with game data
        game_features_df: DataFrame with game features
        team_features_df: DataFrame with team features

    Returns:
        Processed DataFrame ready for training
    """
    # Exit early if any dataframe is empty
    if games_df.is_empty() or game_features_df.is_empty():
        return pl.DataFrame()

    # Check if the dataframes have required columns
    required_game_columns = [
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "date",
    ]
    if not all(col in games_df.columns for col in required_game_columns):
        return pl.DataFrame()

    # Add target variables
    games_with_targets = games_df.with_columns(
        [
            (pl.col("home_score") > pl.col("away_score")).alias("home_win"),
            (pl.col("home_score") - pl.col("away_score")).alias("point_diff"),
        ]
    )

    # Pivot game features to get one row per game with all features as columns
    if not game_features_df.is_empty():
        # Ensure game_features_df has required columns
        required_feature_columns = ["game_id", "feature_name", "feature_value"]
        if all(col in game_features_df.columns for col in required_feature_columns):
            # Pivot the features
            game_features_pivot = game_features_df.select(
                ["game_id", "feature_name", "feature_value"]
            ).pivot(
                index="game_id",
                columns="feature_name",
                values="feature_value",
                aggregate_function="first",
            )

            # Join with games data
            training_data = games_with_targets.join(game_features_pivot, on="game_id", how="inner")
        else:
            # If game features don't have required columns, just use game data
            training_data = games_with_targets
    else:
        # If no game features, just use game data
        training_data = games_with_targets

    # Convert all columns to appropriate types
    # Float columns for features, integer for scores, boolean for win
    training_data = training_data.with_columns([pl.col("point_diff").cast(pl.Float64)])

    # For any columns that are of object type, attempt to convert to float
    for col in training_data.columns:
        if col not in ["game_id", "home_team", "away_team", "date", "home_win"]:
            if training_data[col].dtype == pl.Object:
                training_data = training_data.with_columns([pl.col(col).cast(pl.Float64)])

    return training_data


def split_data(
    data: pl.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data into training, validation, and test sets.

    Args:
        data: DataFrame to split
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Check if ratios sum to approximately 1.0
    if not 0.99 <= train_ratio + val_ratio + test_ratio <= 1.01:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Get all unique game IDs
    game_ids = data["game_id"].unique().to_list()

    # Shuffle the game IDs to ensure random splitting
    random.shuffle(game_ids)

    # Calculate indices for splitting
    train_end = int(len(game_ids) * train_ratio)
    val_end = train_end + int(len(game_ids) * val_ratio)

    # Split game IDs
    train_ids = game_ids[:train_end]
    val_ids = game_ids[train_end:val_end]
    test_ids = game_ids[val_end:]

    # Filter data based on game IDs
    train_data = data.filter(pl.col("game_id").is_in(train_ids))
    val_data = data.filter(pl.col("game_id").is_in(val_ids))
    test_data = data.filter(pl.col("game_id").is_in(test_ids))

    return train_data, val_data, test_data
