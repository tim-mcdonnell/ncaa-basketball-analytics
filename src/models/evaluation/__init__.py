"""
Model evaluation functionality.
"""

from .metrics import (
    calculate_accuracy,
    calculate_point_spread_accuracy,
    calculate_calibration_metrics,
    calculate_feature_importance,
)
from .cross_validation import (
    perform_cross_validation,
    create_time_series_splits,
    create_kfold_splits,
    aggregate_cv_results,
)
from .visualization import (
    plot_learning_curves,
    plot_feature_importance,
    plot_calibration_curve,
    plot_confusion_matrix,
    save_evaluation_plots,
)
from src.models.evaluation.evaluate import evaluate_model

__all__ = [
    # Metrics
    "calculate_accuracy",
    "calculate_point_spread_accuracy",
    "calculate_calibration_metrics",
    "calculate_feature_importance",
    # Cross-validation
    "perform_cross_validation",
    "create_time_series_splits",
    "create_kfold_splits",
    "aggregate_cv_results",
    # Visualization
    "plot_learning_curves",
    "plot_feature_importance",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "save_evaluation_plots",
    "evaluate_model",
]
