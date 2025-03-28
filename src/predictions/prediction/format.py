"""
Module for formatting prediction results for display or API responses.
"""

import logging
from typing import Dict, List, Optional, Any
from fractions import Fraction

import polars as pl


def format_predictions(
    predictions: pl.DataFrame,
    confidence_thresholds: Optional[Dict[str, float]] = None,
    include_confidence: Optional[List[str]] = None,
    probability_format: str = "percentage",
    sort_by: str = "confidence",
    sort_ascending: bool = False,
) -> List[Dict[str, Any]]:
    """
    Format raw prediction results into a more presentable format.

    Args:
        predictions: DataFrame containing raw prediction results
        confidence_thresholds: Dictionary mapping confidence levels to probability thresholds
            Default: {"Low": 0.0, "Medium": 0.6, "High": 0.75}
        include_confidence: List of confidence levels to include (e.g., ["High", "Medium"])
            Default: include all confidence levels
        probability_format: Format for win probability ("percentage", "decimal", or "fraction")
        sort_by: Column to sort results by ("confidence", "game_id", etc.)
        sort_ascending: Whether to sort in ascending order

    Returns:
        List of dictionaries containing formatted prediction results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Formatting {len(predictions)} predictions")

    # Handle empty DataFrame
    if len(predictions) == 0:
        return []

    # Set default confidence thresholds if not provided
    if confidence_thresholds is None:
        confidence_thresholds = {"Low": 0.0, "Medium": 0.6, "High": 0.75}

    # Convert predictions to a pandas DataFrame for easier manipulation
    predictions_pd = predictions.to_pandas()

    # Add formatted win probability
    formatted_probabilities = []
    for prob in predictions_pd["win_probability"]:
        if probability_format == "percentage":
            formatted_probabilities.append(f"{int(prob * 100)}%")
        elif probability_format == "decimal":
            formatted_probabilities.append(
                f"{prob:.2f}".rstrip("0").rstrip(".") if "." in f"{prob:.2f}" else f"{prob:.2f}"
            )
        elif probability_format == "fraction":
            fraction = Fraction(prob).limit_denominator(10)
            formatted_probabilities.append(f"{fraction.numerator}/{fraction.denominator}")
        else:
            formatted_probabilities.append(f"{prob:.2f}")

    predictions_pd["win_probability_formatted"] = formatted_probabilities

    # Add confidence level
    def get_confidence_level(prob):
        for level, threshold in sorted(
            confidence_thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if prob >= threshold:
                return level
        return list(confidence_thresholds.keys())[0]  # Return lowest confidence level as fallback

    predictions_pd["confidence_level"] = predictions_pd["win_probability"].apply(
        get_confidence_level
    )

    # Filter by confidence level if requested
    if include_confidence:
        predictions_pd = predictions_pd[predictions_pd["confidence_level"].isin(include_confidence)]

    # Sort results
    if sort_by == "confidence":
        predictions_pd.sort_values("win_probability", ascending=sort_ascending, inplace=True)
    else:
        if sort_by in predictions_pd.columns:
            predictions_pd.sort_values(sort_by, ascending=sort_ascending, inplace=True)
        else:
            logger.warning(
                f"Sort column '{sort_by}' not found in predictions DataFrame. Using default sorting."
            )

    # Convert to list of dictionaries
    result = predictions_pd.to_dict(orient="records")

    logger.info(f"Formatted {len(result)} predictions")
    return result


def calculate_prediction_accuracy(
    predictions: pl.DataFrame, actual_results: pl.DataFrame
) -> Dict[str, Any]:
    """
    Calculate accuracy of predictions compared to actual results.

    Args:
        predictions: DataFrame containing predictions
        actual_results: DataFrame containing actual game results

    Returns:
        Dictionary with accuracy metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Calculating prediction accuracy")

    # Merge predictions with actual results
    merged = predictions.join(actual_results, left_on="game_id", right_on="game_id", how="inner")

    if len(merged) == 0:
        logger.warning("No matching games found between predictions and actual results")
        return {"accuracy": 0.0, "games_evaluated": 0, "correct_predictions": 0}

    # Count correct predictions
    correct_predictions = merged.filter(pl.col("predicted_winner") == pl.col("winning_team"))

    accuracy = len(correct_predictions) / len(merged)

    # Calculate accuracy by confidence level
    confidence_levels = (
        merged["confidence_level"].unique().to_list()
        if "confidence_level" in merged.columns
        else []
    )
    accuracy_by_confidence = {}

    for level in confidence_levels:
        level_games = merged.filter(pl.col("confidence_level") == level)
        level_correct = level_games.filter(pl.col("predicted_winner") == pl.col("winning_team"))
        level_accuracy = len(level_correct) / len(level_games) if len(level_games) > 0 else 0.0
        accuracy_by_confidence[level] = {
            "accuracy": level_accuracy,
            "games": len(level_games),
            "correct": len(level_correct),
        }

    logger.info(
        f"Overall prediction accuracy: {accuracy:.2f} ({len(correct_predictions)}/{len(merged)} correct)"
    )

    return {
        "accuracy": accuracy,
        "games_evaluated": len(merged),
        "correct_predictions": len(correct_predictions),
        "accuracy_by_confidence": accuracy_by_confidence,
    }
