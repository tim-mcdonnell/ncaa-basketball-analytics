"""
Prediction functionality for NCAA basketball games.
"""

from src.predictions.prediction.generate import generate_predictions
from src.predictions.prediction.format import format_predictions, calculate_prediction_accuracy

__all__ = ["generate_predictions", "format_predictions", "calculate_prediction_accuracy"]
