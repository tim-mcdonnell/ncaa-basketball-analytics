import pytest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import patch, MagicMock
from sklearn.calibration import calibration_curve
import pandas as pd

from src.models.evaluation.visualization import (
    plot_learning_curves,
    plot_feature_importance,
    plot_calibration_curve,
    plot_confusion_matrix,
    save_evaluation_plots,
)


class TestVisualizationFunctions:
    """Test suite for model evaluation visualization functions."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample training metrics data as a single dictionary."""
        # Training data over 10 epochs
        epochs = list(range(1, 11))
        metrics_data = {
            "epoch": epochs,
            "train_loss": [2.5, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
            "val_loss": [2.6, 2.3, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.25, 1.2],
            "train_accuracy": [0.60, 0.65, 0.70, 0.73, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86],
            "val_accuracy": [0.58, 0.63, 0.68, 0.71, 0.74, 0.76, 0.77, 0.78, 0.79, 0.80],
            "train_rmse": [0.9, 0.85, 0.8, 0.75, 0.7, 0.68, 0.65, 0.63, 0.61, 0.60],
            "val_rmse": [0.92, 0.87, 0.82, 0.78, 0.74, 0.72, 0.70, 0.68, 0.67, 0.65],
        }
        return metrics_data

    @pytest.fixture
    def sample_feature_importance(self):
        """Create sample feature importance data."""
        # Return unsorted dictionary, function should sort internally
        return {
            "team_win_ratio": 0.25,
            "point_differential": 0.20,
            "offensive_rating": 0.15,
            "defensive_rating": 0.15,
            "turnover_percentage": 0.10,
            "rebound_percentage": 0.10,
            "free_throw_rate": 0.05,
        }

    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data for calibration curve."""
        # 100 samples
        predictions = torch.tensor(np.linspace(0.1, 0.9, 100))

        # Actuals - higher prediction means higher chance of being actual 1
        actuals = torch.zeros_like(predictions)
        actuals[predictions > 0.7] = 1.0

        # Add some randomness in middle range
        middle_mask = (predictions > 0.3) & (predictions < 0.7)
        middle_count = middle_mask.sum().item()
        actuals[middle_mask] = torch.tensor(
            np.random.choice([0.0, 1.0], size=middle_count, p=[0.7, 0.3])  # Slight bias toward 0
        )

        return predictions.numpy(), actuals.numpy()  # Return numpy arrays

    @pytest.fixture
    def sample_confusion(self):
        """Create sample confusion matrix."""
        # Create a 2x2 confusion matrix
        # Format: [[TP, FP], [FN, TN]] -> This looks wrong, should be [[TN, FP], [FN, TP]] or check function
        # Let's assume sklearn standard: [[TN, FP], [FN, TP]]
        # return np.array([
        #     [85, 10], # True Neg, False Pos
        #     [15, 90]  # False Neg, True Pos
        # ])
        # Sticking with original test data [[85, 10], [15, 90]] and assume function expects this
        return np.array([[85, 10], [15, 90]])

    def test_plot_learning_curves(self, sample_metrics):
        """Test plotting learning curves."""
        # Mock the plotting functions
        with patch("src.models.evaluation.visualization.plt") as mock_plt:
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_plt.subplots.return_value = (
                mock_fig_instance,
                MagicMock(),
            )  # subplots returns fig, ax
            _mock_ax = mock_fig_instance.add_subplot.return_value  # Prefix unused variable

            # Call the function with the single metrics dict
            _fig = plot_learning_curves(metrics=sample_metrics)

            # Verify figure creation
            mock_plt.subplots.assert_called_once()

            # Verify plot calls - train/val loss + 2 train/val metrics = 6 lines
            num_metrics_plotted = len([k for k in sample_metrics if k != "epoch"])
            ax = mock_plt.subplots.return_value[1]  # Get the mocked Axes object
            assert (
                ax.plot.call_count == num_metrics_plotted
            ), f"Should plot {num_metrics_plotted} lines"

            # Verify labels and legends
            assert ax.set_xlabel.call_count == 1, "Should set x label once"
            assert ax.set_ylabel.call_count == 1, "Should set y label once"
            assert ax.legend.call_count == 1, "Should create legend once"
            assert ax.set_title.call_count == 1, "Should set title once"

    def test_plot_feature_importance(self, sample_feature_importance):
        """Test plotting feature importance."""
        feature_names = list(sample_feature_importance.keys())
        importance_scores = list(sample_feature_importance.values())

        # Mock the plotting functions
        with patch("src.models.evaluation.visualization.plt") as mock_plt:
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_plt.subplots.return_value = (mock_fig_instance, MagicMock())
            ax = mock_plt.subplots.return_value[1]  # Get the mocked Axes object

            # Call the function with separate lists
            _fig = plot_feature_importance(
                feature_names=feature_names, importance_scores=importance_scores
            )

            # Verify figure creation
            mock_plt.subplots.assert_called_once()

            # Verify barplot creation
            ax.barh.assert_called_once()

            # Verify the number of features plotted (function sorts and might take top_n)
            # Get the sorted data as the function would process it
            df = pd.DataFrame({"Feature": feature_names, "Importance": importance_scores})
            df = df.sort_values("Importance", ascending=True).head(20)  # Default top_n is 20

            call_args, call_kwargs = ax.barh.call_args
            plotted_features = call_args[0]  # First positional arg is y (feature names)
            plotted_scores = call_args[1]  # Second positional arg is width (scores)

            assert len(plotted_features) == len(df), f"Should plot {len(df)} features"
            assert len(plotted_scores) == len(df), f"Should plot {len(df)} scores"

            # Verify sorting - compare the whole list of plotted features
            # Convert plotted_features (likely a Series) to list for comparison
            assert (
                plotted_features.tolist() == df["Feature"].tolist()
            ), f"Plotted features order mismatch. Expected {df['Feature'].tolist()}, got {plotted_features.tolist()}"

            # Verify title and labels
            assert ax.set_title.call_count == 1
            assert ax.set_xlabel.call_count == 1
            assert ax.set_ylabel.call_count == 1

    def test_plot_calibration_curve(self, sample_predictions):
        """Test plotting calibration curve."""
        predictions, actuals = sample_predictions
        prob_true, prob_pred = calibration_curve(actuals, predictions, n_bins=10)

        # Mock the plotting functions
        with patch("src.models.evaluation.visualization.plt") as mock_plt:
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_plt.subplots.return_value = (mock_fig_instance, MagicMock())
            ax = mock_plt.subplots.return_value[1]  # Get the mocked Axes object

            # Call the function with calculated probabilities as lists
            _fig = plot_calibration_curve(
                prob_true=prob_true.tolist(), prob_pred=prob_pred.tolist()
            )

            # Verify figure creation
            mock_plt.subplots.assert_called_once()

            # Verify plot and reference line
            assert ax.plot.call_count == 2, "Should plot 2 lines (calibration curve and reference)"

            # Verify labels and title
            ax.set_xlabel.assert_called_once()
            ax.set_ylabel.assert_called_once()
            ax.set_title.assert_called_once()
            ax.legend.assert_called_once()  # Check legend is called

    def test_plot_confusion_matrix(self, sample_confusion):
        """Test plotting confusion matrix."""
        # Mock the plotting functions
        with (
            patch("src.models.evaluation.visualization.plt") as mock_plt,
            patch("src.models.evaluation.visualization.sns") as mock_sns,
        ):
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_plt.subplots.return_value = (mock_fig_instance, MagicMock())
            ax = mock_plt.subplots.return_value[1]  # Get the mocked Axes object

            # Call the function with numpy array converted to list and correct arg name
            _fig = plot_confusion_matrix(
                conf_matrix=sample_confusion.tolist(),  # Convert to list
                labels=["Win", "Loss"],  # Use 'labels' arg
            )

            # Verify figure creation
            mock_plt.subplots.assert_called_once()

            # Verify heatmap creation
            mock_sns.heatmap.assert_called_once()

            # Verify correct confusion matrix (as list) and ax were passed to heatmap
            call_args, call_kwargs = mock_sns.heatmap.call_args
            passed_matrix = call_args[0]
            assert passed_matrix == sample_confusion.tolist()
            assert call_kwargs.get("ax") == ax

            # Verify labels and title
            ax.set_xlabel.assert_called_once()
            ax.set_ylabel.assert_called_once()
            ax.set_title.assert_called_once()

    def test_save_evaluation_plots(
        self, sample_metrics, sample_feature_importance, sample_confusion, sample_predictions
    ):
        """Test saving all evaluation plots."""
        # Prepare data in the expected format for the plotting functions that save_evaluation_plots calls
        feature_names = list(sample_feature_importance.keys())
        importance_scores = list(sample_feature_importance.values())
        predictions_np, actuals_np = sample_predictions
        prob_true, prob_pred = calibration_curve(actuals_np, predictions_np, n_bins=10)
        conf_matrix_list = sample_confusion.tolist()

        # Create mock figures
        mock_figures = {
            "learning_curves": MagicMock(),
            "feature_importance": MagicMock(),
            "calibration_curve": MagicMock(),
            "confusion_matrix": MagicMock(),
        }

        # Mock the individual plotting functions used by save_evaluation_plots
        with (
            patch(
                "src.models.evaluation.visualization.plot_learning_curves",
                return_value=mock_figures["learning_curves"],
            ) as mock_plot_lc,
            patch(
                "src.models.evaluation.visualization.plot_feature_importance",
                return_value=mock_figures["feature_importance"],
            ) as mock_plot_fi,
            patch(
                "src.models.evaluation.visualization.plot_calibration_curve",
                return_value=mock_figures["calibration_curve"],
            ) as mock_plot_cc,
            patch(
                "src.models.evaluation.visualization.plot_confusion_matrix",
                return_value=mock_figures["confusion_matrix"],
            ) as mock_plot_cm,
        ):
            # Create a temporary directory for saving
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Prepare the metrics dictionary expected by save_evaluation_plots
                metrics_for_saving = {
                    "learning_curve_data": sample_metrics,  # Nested dict for learning curves
                    "feature_names": feature_names,  # Top-level feature names
                    "importance_scores": importance_scores,  # Top-level scores
                    "calibration_curve": {  # Nested dict for calibration
                        "prob_true": prob_true.tolist(),
                        "prob_pred": prob_pred.tolist(),
                    },
                    "confusion_matrix": conf_matrix_list,  # Top-level matrix list
                    # 'labels': ["Win", "Loss"] # Labels handled by plot_confusion_matrix default
                }

                # Call the function to save plots
                save_paths = save_evaluation_plots(
                    metrics=metrics_for_saving, output_dir=tmp_dir, model_name="test_model"
                )

                # Verify that the mocked plotting functions were called correctly
                mock_plot_lc.assert_called_once()
                mock_plot_fi.assert_called_once()
                mock_plot_cc.assert_called_once()
                mock_plot_cm.assert_called_once()

                # Verify that the figures returned by mocks were saved
                # Check args passed to plot_learning_curves (positional arg 0)
                lc_args, lc_kwargs = mock_plot_lc.call_args
                assert lc_args[0] == metrics_for_saving["learning_curve_data"]
                # assert lc_kwargs.get('metrics') == metrics_for_saving['learning_curve_data']
                assert os.path.dirname(lc_kwargs.get("save_path")) == tmp_dir

                # Check args passed to plot_feature_importance (keyword args)
                fi_args, fi_kwargs = mock_plot_fi.call_args
                assert fi_kwargs.get("feature_names") == metrics_for_saving["feature_names"]
                assert fi_kwargs.get("importance_scores") == metrics_for_saving["importance_scores"]
                assert os.path.dirname(fi_kwargs.get("save_path")) == tmp_dir

                # Check args passed to plot_calibration_curve (keyword args)
                cc_args, cc_kwargs = mock_plot_cc.call_args
                assert (
                    cc_kwargs.get("prob_true")
                    == metrics_for_saving["calibration_curve"]["prob_true"]
                )
                assert (
                    cc_kwargs.get("prob_pred")
                    == metrics_for_saving["calibration_curve"]["prob_pred"]
                )
                assert os.path.dirname(cc_kwargs.get("save_path")) == tmp_dir

                # Check args passed to plot_confusion_matrix (keyword args)
                cm_args, cm_kwargs = mock_plot_cm.call_args
                assert cm_kwargs.get("conf_matrix") == metrics_for_saving["confusion_matrix"]
                # assert cm_kwargs.get('labels') == metrics_for_saving['labels'] # Uses default
                assert os.path.dirname(cm_kwargs.get("save_path")) == tmp_dir

                # Verify saved paths dictionary
                assert len(save_paths) == 4, "Should return paths for 4 plots"
                for plot_type, path in save_paths.items():
                    # plot_type will be 'calibration', not 'calibration_curve'
                    assert plot_type in [
                        "learning_curves",
                        "feature_importance",
                        "calibration",
                        "confusion_matrix",
                    ]
                    # assert os.path.exists(path), f"Plot file should exist: {path}" # Remove: Mocks don't save files
                    assert os.path.basename(path).startswith("test_model_")
                    assert os.path.basename(path).endswith(".png")
