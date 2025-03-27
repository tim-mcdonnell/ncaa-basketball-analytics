"""Machine learning model configuration model."""

from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator

from src.config.base import BaseConfig


class ModelHyperparameters(BaseConfig):
    """Hyperparameters for machine learning models.

    This model defines hyperparameters for training machine learning models.

    Example YAML configuration:
    ```yaml
    model:
      hyperparameters:
        learning_rate: 0.01
        max_depth: 5
        num_estimators: 100
        dropout: 0.2
    ```
    """

    learning_rate: Optional[float] = Field(
        default=0.01, description="Learning rate for gradient-based optimization", gt=0
    )
    max_depth: Optional[int] = Field(
        default=None, description="Maximum depth for tree-based models", ge=1
    )
    num_estimators: Optional[int] = Field(
        default=100, description="Number of estimators for ensemble models", ge=1
    )
    dropout: Optional[float] = Field(
        default=None, description="Dropout rate for neural networks", ge=0, le=1
    )
    hidden_layers: Optional[List[int]] = Field(
        default=None, description="Hidden layer sizes for neural networks"
    )
    batch_size: Optional[int] = Field(default=32, description="Batch size for training", ge=1)
    epochs: Optional[int] = Field(default=10, description="Number of epochs for training", ge=1)

    # Add additional hyperparameters as needed
    # This model can be extended with specific hyperparameters for different model types


class ModelConfig(BaseConfig):
    """Machine learning model configuration.

    This model defines configuration for training and using machine learning models.

    Example YAML configuration:
    ```yaml
    model:
      type: "xgboost"
      feature_set: "all_features"
      target: "win_probability"
      hyperparameters:
        learning_rate: 0.01
        max_depth: 5
        num_estimators: 100
      evaluation:
        metrics: ["accuracy", "f1_score", "roc_auc"]
        validation_split: 0.2
        cross_validation_folds: 5
      experiment_tracking:
        enabled: true
        log_artifacts: true
    ```
    """

    type: str = Field(description="Type of machine learning model to use")
    feature_set: str = Field(description="Name of the feature set to use for training/prediction")
    target: str = Field(description="Name of the target variable to predict")
    hyperparameters: ModelHyperparameters = Field(
        default_factory=ModelHyperparameters, description="Model hyperparameters"
    )
    evaluation: Dict[str, Union[List[str], float, int]] = Field(
        default_factory=lambda: {
            "metrics": ["accuracy", "f1_score"],
            "validation_split": 0.2,
            "cross_validation_folds": 5,
        },
        description="Model evaluation settings",
    )
    experiment_tracking: Dict[str, bool] = Field(
        default_factory=lambda: {"enabled": True, "log_artifacts": True},
        description="Experiment tracking settings for MLflow",
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility", ge=0)

    @field_validator("type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate that the model type is supported."""
        valid_types = ["linear", "logistic", "xgboost", "lightgbm", "neural_network"]
        if v.lower() not in [t.lower() for t in valid_types]:
            raise ValueError(f"Model type '{v}' is not supported. Valid types: {valid_types}")
        return v
