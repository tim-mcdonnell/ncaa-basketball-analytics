import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from matplotlib.figure import Figure
import seaborn as sns


def plot_learning_curves(
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Learning Curves',
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot learning curves from training metrics.
    
    Args:
        metrics: Dictionary with metric names and their values per epoch
        figsize: Figure size
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot train/val loss
    if 'train_loss' in metrics and 'val_loss' in metrics:
        epochs = range(1, len(metrics['train_loss']) + 1)
        ax.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
        ax.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    
    # Plot other metrics if available
    for name, values in metrics.items():
        if name not in ['train_loss', 'val_loss']:
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=name)
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: List[float],
    importance_std_devs: Optional[List[float]] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Feature Importance',
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot feature importance.
    
    Args:
        feature_names: Names of features
        importance_scores: Importance scores for each feature
        importance_std_devs: Standard deviations of importance scores
        top_n: Number of top features to show
        figsize: Figure size
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    # Limit to top N features
    if len(feature_names) > top_n:
        indices = np.argsort(importance_scores)[-top_n:]
        feature_names = [feature_names[i] for i in indices]
        importance_scores = [importance_scores[i] for i in indices]
        if importance_std_devs:
            importance_std_devs = [importance_std_devs[i] for i in indices]
    
    # Create DataFrame for better plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    df = df.sort_values('Importance', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    bars = ax.barh(df['Feature'], df['Importance'], color='skyblue')
    
    # Add error bars if provided
    if importance_std_devs:
        error_positions = [(bar.get_y() + bar.get_height() / 2) for bar in bars]
        ax.errorbar(
            df['Importance'],
            error_positions,
            xerr=importance_std_devs,
            fmt='none',
            ecolor='black',
            capsize=3
        )
    
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    
    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(
    prob_true: List[float],
    prob_pred: List[float],
    figsize: Tuple[int, int] = (8, 8),
    title: str = 'Calibration Curve',
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot calibration curve.
    
    Args:
        prob_true: True probabilities
        prob_pred: Predicted probabilities
        figsize: Figure size
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the calibration curve
    ax.plot(prob_pred, prob_true, 's-', label='Calibration Curve')
    
    # Plot the ideal calibration curve (diagonal)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfectly Calibrated')
    
    ax.set_title(title)
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set equal aspect for better visualization
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    conf_matrix: List[List[int]],
    labels: List[str] = ['Away Win', 'Home Win'],
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix as a 2D list
        labels: Labels for classes
        figsize: Figure size
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_evaluation_plots(
    metrics: Dict[str, Any],
    output_dir: str,
    model_name: str = 'model',
    figsize: Tuple[int, int] = (10, 6)
) -> Dict[str, str]:
    """
    Save all evaluation plots to files.
    
    Args:
        metrics: Dictionary with metrics and data for plotting
        output_dir: Directory to save plots
        model_name: Name of the model for filenames
        figsize: Figure size for plots
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = {}
    
    # Plot learning curves if available
    if 'learning_curve_data' in metrics:
        learning_curve_path = os.path.join(output_dir, f'{model_name}_learning_curves.png')
        plot_learning_curves(
            metrics['learning_curve_data'],
            figsize=figsize,
            title=f'{model_name} Learning Curves',
            save_path=learning_curve_path
        )
        saved_plots['learning_curves'] = learning_curve_path
    
    # Plot feature importance if available
    if all(k in metrics for k in ['feature_names', 'importance_scores']):
        importance_path = os.path.join(output_dir, f'{model_name}_feature_importance.png')
        plot_feature_importance(
            feature_names=metrics['feature_names'],
            importance_scores=metrics['importance_scores'],
            importance_std_devs=metrics.get('importance_std_devs'),
            figsize=figsize,
            title=f'{model_name} Feature Importance',
            save_path=importance_path
        )
        saved_plots['feature_importance'] = importance_path
    
    # Plot calibration curve if available
    if 'calibration_curve' in metrics:
        cal_curve = metrics['calibration_curve']
        if 'prob_true' in cal_curve and 'prob_pred' in cal_curve:
            calibration_path = os.path.join(output_dir, f'{model_name}_calibration.png')
            plot_calibration_curve(
                prob_true=cal_curve['prob_true'],
                prob_pred=cal_curve['prob_pred'],
                figsize=figsize,
                title=f'{model_name} Calibration Curve',
                save_path=calibration_path
            )
            saved_plots['calibration'] = calibration_path
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in metrics:
        confusion_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
        plot_confusion_matrix(
            conf_matrix=metrics['confusion_matrix'],
            figsize=figsize,
            title=f'{model_name} Confusion Matrix',
            save_path=confusion_path
        )
        saved_plots['confusion_matrix'] = confusion_path
    
    return saved_plots 