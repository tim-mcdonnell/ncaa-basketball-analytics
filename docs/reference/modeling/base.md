---
title: Base Modeling Components
description: API reference for base model classes and configurations.
---

# Base Modeling Components (`src.models.base`)

This module provides the fundamental building blocks for all predictive models in the system.

## `ModelConfig`

::: src.models.base.ModelConfig
    options:
      show_root_heading: true
      heading_level: 2

## `ModelVersion`

::: src.models.base.ModelVersion
    options:
      show_root_heading: true
      heading_level: 2

## `BaseModel`

::: src.models.base.BaseModel
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - __init__
        - forward
        - predict
        - save
        - load
        - get_version
        - get_hyperparameters
