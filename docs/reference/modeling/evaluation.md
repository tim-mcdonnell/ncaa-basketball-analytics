---
title: Model Evaluation
description: API reference for model evaluation metrics and visualization.
---

# Model Evaluation (`src.models.evaluation`)

This module provides tools for evaluating model performance and visualizing results.

## Metrics (`src.models.evaluation.metrics`)

### Functions

::: src.models.evaluation.metrics.calculate_prediction_accuracy
    options:
      show_root_heading: true
      heading_level: 3

::: src.models.evaluation.metrics.calculate_point_spread_accuracy
    options:
      show_root_heading: true
      heading_level: 3

::: src.models.evaluation.metrics.calculate_calibration_metrics
    options:
      show_root_heading: true
      heading_level: 3

::: src.models.evaluation.metrics.calculate_feature_importance
    options:
      show_root_heading: true
      heading_level: 3

::: src.models.evaluation.metrics.calculate_accuracy
    options:
      show_root_heading: true
      heading_level: 3

### Classes

::: src.models.evaluation.metrics.EvaluationMetrics
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - calculate_all_metrics
        - get_report

## Visualization (`src.models.evaluation.visualization`)

### Functions

::: src.models.evaluation.visualization.plot_learning_curves
    options:
      show_root_heading: true
      heading_level: 3

::: src.models.evaluation.visualization.plot_feature_importance
    options:
      show_root_heading: true
      heading_level: 3

::: src.models.evaluation.visualization.plot_calibration_curve
    options:
      show_root_heading: true
      heading_level: 3

::: src.models.evaluation.visualization.plot_confusion_matrix
    options:
      show_root_heading: true
      heading_level: 3

::: src.models.evaluation.visualization.save_evaluation_plots
    options:
      show_root_heading: true
      heading_level: 3
