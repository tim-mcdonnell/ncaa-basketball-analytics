import numpy as np
import polars as pl
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
)
from datetime import datetime


def calculate_prediction_accuracy(
    predictions: Union[np.ndarray, torch.Tensor, List[float]],
    targets: Union[np.ndarray, torch.Tensor, List[float]],
    threshold: float = 0.5,
    return_confusion_matrix: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Calculate binary prediction accuracy and optionally confusion matrix.

    Args:
        predictions: Predicted probabilities
        targets: True binary labels
        threshold: Threshold for converting probabilities to binary predictions
        return_confusion_matrix: Whether to return confusion matrix along with accuracy

    Returns:
        If return_confusion_matrix is False: Accuracy (float)
        If return_confusion_matrix is True: Tuple of (accuracy, confusion_matrix)
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()

    # Ensure arrays are 1D
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # Convert to binary predictions using threshold
    binary_predictions = (predictions > threshold).astype(int)
    binary_targets = (targets > threshold).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(binary_targets, binary_predictions)

    # Return just accuracy if not requesting confusion matrix
    if not return_confusion_matrix:
        return accuracy

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(binary_targets, binary_predictions)

    return accuracy, conf_matrix


def calculate_point_spread_accuracy(
    y_true: Union[np.ndarray, torch.Tensor, List[float]],
    y_pred: Union[np.ndarray, torch.Tensor, List[float]],
    return_details: bool = False,
) -> Union[float, Tuple[float, List[Dict[str, Any]]]]:
    """
    Calculate accuracy specifically for point spread predictions.

    Args:
        y_true: True point spreads
        y_pred: Predicted point spreads
        return_details: Whether to return detailed metrics for each prediction

    Returns:
        If return_details is False: Winner prediction accuracy (float)
        If return_details is True: Tuple of (accuracy, list of detailed results)
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    # Ensure arrays are 1D
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate if the prediction correctly indicated the winner
    correct_winner = (y_true > 0) == (y_pred > 0)
    winner_accuracy = float(np.mean(correct_winner))

    if not return_details:
        return winner_accuracy

    # Calculate detailed metrics for each prediction
    details = []
    for i in range(len(y_true)):
        details.append(
            {
                "correct_winner": bool(correct_winner[i]),  # Convert numpy bool to Python bool
                "prediction_error": float(abs(y_true[i] - y_pred[i])),
                "actual_spread": float(y_true[i]),
                "predicted_spread": float(y_pred[i]),
            }
        )

    return winner_accuracy, details


def manual_calibration_curve(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manual implementation of calibration curve calculation.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Tuple of (prob_true, prob_pred) arrays
    """
    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1

    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = np.zeros(len(bins) - 1)
    prob_pred = np.zeros(len(bins) - 1)

    prob_true[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred


def calculate_calibration_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List[float]],
    y_pred: Union[np.ndarray, torch.Tensor, List[float]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculate calibration metrics for probabilistic predictions.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics and curves
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    # Ensure arrays are 1D
    y_true = np.array(y_true).flatten()
    y_true_binary = (y_true > 0).astype(int)  # Convert to binary
    y_pred = np.array(y_pred).flatten()

    # Normalize predictions to [0, 1] if needed
    if np.min(y_pred) < 0 or np.max(y_pred) > 1:
        # Simple min-max scaling
        y_pred_norm = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
    else:
        y_pred_norm = y_pred

    try:
        # Calculate Brier score
        brier_score = brier_score_loss(y_true_binary, y_pred_norm)

        # Calculate calibration curve using manual implementation
        prob_true, prob_pred = manual_calibration_curve(y_true_binary, y_pred_norm, n_bins=n_bins)

        # Calculate calibration error
        calib_error = np.mean(np.abs(prob_true - prob_pred))

        # Calculate reliability and resolution
        mean_obs = np.mean(y_true_binary)
        reliability = np.sum(
            bin_weights := np.bincount(
                np.digitize(y_pred_norm, np.linspace(0, 1, n_bins + 1)) - 1, minlength=n_bins
            )
            / len(y_pred_norm)
            * (prob_pred - prob_true) ** 2
        )
        resolution = np.sum(bin_weights * (prob_true - mean_obs) ** 2)

        # Create bins information for reporting
        bins = []
        for i in range(n_bins):
            bin_start = i / n_bins
            bin_end = (i + 1) / n_bins
            bin_mask = (y_pred_norm >= bin_start) & (y_pred_norm < bin_end)

            # Always create a bin even if no samples
            if np.any(bin_mask):
                bin_info = {
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "pred_mean": float(np.mean(y_pred_norm[bin_mask])),
                    "actual_mean": float(np.mean(y_true_binary[bin_mask])),
                    "samples": int(np.sum(bin_mask)),
                }
            else:
                # Create a default bin with no samples
                bin_info = {
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "pred_mean": float((bin_start + bin_end) / 2),
                    "actual_mean": float(mean_obs),  # Use global mean as default
                    "samples": 0,
                }
            bins.append(bin_info)

        # Generate confusion matrix
        y_pred_binary = (y_pred_norm > 0.5).astype(int)
        conf_matrix = confusion_matrix(y_true_binary, y_pred_binary).tolist()

        return {
            "brier_score": brier_score,
            "calibration_error": calib_error,
            "reliability": reliability,
            "resolution": resolution,
            "calibration_curve": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
            "bins": bins,
            "confusion_matrix": conf_matrix,
        }
    except Exception as e:
        # Return empty metrics if calculation fails
        empty_bins = [
            {
                "bin_start": i / n_bins,
                "bin_end": (i + 1) / n_bins,
                "pred_mean": 0.5,
                "actual_mean": 0.5,
                "samples": 0,
            }
            for i in range(n_bins)
        ]

        return {
            "brier_score": None,
            "calibration_error": None,
            "reliability": None,
            "resolution": None,
            "calibration_curve": {"prob_true": [], "prob_pred": []},
            "bins": empty_bins,
            "confusion_matrix": [[0, 0], [0, 0]],
            "error": str(e),
        }


def calculate_feature_importance(
    model: Any,
    features: Union[np.ndarray, torch.Tensor, pl.DataFrame],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate feature importance for a predictive model.

    Args:
        model: Model with a predict method
        features: Feature data for importance calculation
        feature_names: Names of features (optional)

    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Handle different input types
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().detach().numpy()
    elif isinstance(features, pl.DataFrame):
        # Extract feature names if not provided
        if feature_names is None:
            feature_names = features.columns
        features_np = features.to_numpy()
    else:
        features_np = np.array(features)

    # Generate default feature names if needed
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(features_np.shape[1])]

    # Check if the model has a built-in feature importance method
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
    else:
        # For PyTorch models, use a simple feature perturbation approach
        class ModelWrapper:
            def __init__(self, torch_model):
                self.model = torch_model

            def predict(self, X):
                if isinstance(X, np.ndarray):
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                else:
                    X_tensor = X
                with torch.no_grad():
                    return self.model.predict(X_tensor).cpu().numpy()

        # Calculate baseline prediction
        baseline_pred = model.predict(torch.tensor(features_np, dtype=torch.float32))
        if isinstance(baseline_pred, torch.Tensor):
            baseline_pred = baseline_pred.cpu().detach().numpy()

        # Calculate importance by perturbing each feature
        importances = np.zeros(features_np.shape[1])
        for i in range(features_np.shape[1]):
            # Create copy with permuted feature
            permuted = features_np.copy()
            permuted[:, i] = np.random.permutation(permuted[:, i])

            # Get predictions for permuted data
            permuted_pred = model.predict(torch.tensor(permuted, dtype=torch.float32))
            if isinstance(permuted_pred, torch.Tensor):
                permuted_pred = permuted_pred.cpu().detach().numpy()

            # Calculate effect of permutation
            feature_effect = np.mean(np.abs(baseline_pred - permuted_pred))
            importances[i] = feature_effect

    # Normalize importances
    if np.sum(importances) > 0:
        importances = importances / np.sum(importances)

    # Create dictionary with feature names
    importance_dict = {
        name: float(importance) for name, importance in zip(feature_names, importances)
    }

    return importance_dict


def calculate_accuracy(
    y_true: Union[np.ndarray, torch.Tensor, List[float]],
    y_pred: Union[np.ndarray, torch.Tensor, List[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate accuracy metrics for binary classification or point spread predictions.

    Args:
        y_true: True labels or values
        y_pred: Predicted probabilities or values
        threshold: Threshold for converting probabilities to binary predictions

    Returns:
        Dictionary with accuracy metrics
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    # Ensure arrays are 1D
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # For regression-type predictions (like point spreads)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # For binary predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_binary = (y_true > 0).astype(int)  # Assuming positive values indicate home team win

    accuracy = accuracy_score(y_true_binary, y_pred_binary)

    # Only calculate these if we have binary targets
    if set(np.unique(y_true_binary)) == {0, 1}:
        try:
            precision = precision_score(y_true_binary, y_pred_binary)
            recall = recall_score(y_true_binary, y_pred_binary)
            # Only calculate ROC AUC if we have probability predictions
            if np.min(y_pred) >= 0 and np.max(y_pred) <= 1:
                roc_auc = roc_auc_score(y_true_binary, y_pred)
            else:
                roc_auc = None
        except Exception:
            precision = None
            recall = None
            roc_auc = None
    else:
        precision = None
        recall = None
        roc_auc = None

    # Create metrics dictionary
    metrics = {"accuracy": accuracy, "mse": mse, "rmse": rmse, "mae": mae}

    # Add metrics that might be None
    if precision is not None:
        metrics["precision"] = precision
    if recall is not None:
        metrics["recall"] = recall
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc

    return metrics


class EvaluationMetrics:
    """
    Class for calculating and aggregating model evaluation metrics.

    This class provides methods to calculate various metrics for model evaluation
    and create comprehensive evaluation reports.
    """

    def __init__(
        self,
        predictions: torch.Tensor,
        actuals: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        feature_names: Optional[List[str]] = None,
        model: Optional[Any] = None,
    ):
        """
        Initialize EvaluationMetrics.

        Args:
            predictions: Model predictions
            actuals: Actual target values
            features: Feature values used for feature importance (optional)
            feature_names: Names of features (optional)
            model: Model for feature importance calculation (optional)
        """
        self.predictions = predictions
        self.actuals = actuals
        self.features = features
        self.feature_names = feature_names
        self.model = model
        self.timestamp = datetime.now().isoformat()

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all available evaluation metrics.

        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}

        # Calculate accuracy metrics
        accuracy_metrics = calculate_accuracy(self.actuals, self.predictions)
        metrics.update(accuracy_metrics)

        # Calculate binary prediction accuracy with confusion matrix
        binary_accuracy, conf_matrix = calculate_prediction_accuracy(
            self.predictions, self.actuals, return_confusion_matrix=True
        )
        metrics["binary_accuracy"] = binary_accuracy
        metrics["confusion_matrix"] = conf_matrix.tolist()

        # Calculate calibration metrics if predictions are probabilities
        try:
            calibration_metrics = calculate_calibration_metrics(
                self.actuals, self.predictions, n_bins=10
            )
            metrics["calibration"] = calibration_metrics
        except Exception as e:
            # Skip if calibration calculation fails
            metrics["calibration_error"] = str(e)

        # Calculate feature importance if features and model provided
        if self.features is not None and self.model is not None:
            try:
                importance = calculate_feature_importance(
                    self.model, self.features, self.feature_names
                )
                metrics["feature_importance"] = importance
            except Exception as e:
                # Skip if feature importance calculation fails
                metrics["feature_importance_error"] = str(e)

        return metrics

    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Returns:
            Dictionary with metrics and metadata
        """
        # Calculate all metrics
        metrics = self.calculate_all_metrics()

        # Create report structure
        report = {
            "metrics": metrics,
            "metadata": {
                "timestamp": self.timestamp,
                "num_samples": len(self.predictions),
                "has_feature_importance": "feature_importance" in metrics,
            },
        }

        return report
