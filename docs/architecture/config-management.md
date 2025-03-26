# Configuration Management

## Overview

This document outlines how configuration is managed in the NCAA Basketball Analytics project using YAML files and Pydantic models. This approach provides both flexibility and type safety.

## Configuration Structure

The project uses hierarchical YAML files stored in the `config/` directory, with Pydantic models for validation and type checking.

## Example Configuration Files

### 1. API Configuration (`config/api_config.yaml`)

```yaml
espn_api:
  base_url: "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
  request_timeout: 30
  rate_limit:
    requests_per_minute: 60
    retry_attempts: 3
    backoff_factor: 2.0
  endpoints:
    teams: "/seasons/{year}/teams"
    events: "/seasons/{year}/events"
    athletes: "/seasons/{year}/athletes"
    rankings: "/seasons/{year}/rankings"
    statistics: "/seasons/{year}/teams/{team_id}/statistics"
    play_by_play: "/events/{event_id}/playbyplay"
  seasons:
    start_year: 2005
    current_year: 2025
```

### 2. Database Configuration (`config/db_config.yaml`)

```yaml
duckdb:
  database_path: "data/ncaa_basketball.duckdb"
  
  # Storage options
  storage:
    parquet_directory: "data/processed/"
    use_compression: true
    compression_method: "zstd"
    
  # Performance tuning
  performance:
    memory_limit: "4GB"
    threads: 4

  # Advanced options
  advanced:
    extension_directory: "extensions/"
    load_extensions:
      - json
      - parquet
      - httpfs
```

### 3. Feature Configuration (`config/feature_config.yaml`)

```yaml
features:
  storage:
    directory: "data/features/"
    format: "parquet"
  
  # Feature computation settings
  computation:
    window_sizes: [1, 3, 5, 10, 15, 30]  # Game windows for rolling features
    recalculation_frequency: "daily"
    
  # Feature groups
  groups:
    team_stats:
      enabled: true
      lookback_periods: [5, 10, 20]
    player_stats:
      enabled: true
      lookback_periods: [5, 10, 20]
    game_context:
      enabled: true
    historical_matchups:
      enabled: true
      max_lookback_years: 3
```

### 4. Model Configuration (`config/model_config.yaml`)

```yaml
models:
  # MLflow settings
  mlflow:
    tracking_uri: "sqlite:///data/models/mlflow.db"
    experiment_name: "ncaa-basketball-predictions"
    model_registry: "data/models/registry"
  
  # Training settings
  training:
    test_size: 0.2
    validation_size: 0.1
    random_state: 42
    cv_folds: 5
    
  # Model-specific parameters
  model_params:
    pytorch:
      architecture: "lstm"
      hidden_layers: [128, 64]
      dropout: 0.2
      learning_rate: 0.001
      batch_size: 64
      epochs: 50
      early_stopping_patience: 5
    lightgbm:
      enabled: true
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 100
      colsample_bytree: 0.8
      subsample: 0.8
      reg_alpha: 0.1
      reg_lambda: 0.1
```

## Pydantic Models Implementation (`src/config/settings.py`)

```python
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, root_validator, validator
import os
from datetime import datetime

class RateLimitConfig(BaseModel):
    requests_per_minute: int = 60
    retry_attempts: int = 3
    backoff_factor: float = 2.0

class APIEndpoints(BaseModel):
    teams: str
    events: str
    athletes: str
    rankings: str
    statistics: str
    play_by_play: str
    
class SeasonConfig(BaseModel):
    start_year: int
    current_year: int = Field(default_factory=lambda: datetime.now().year + (1 if datetime.now().month > 6 else 0))
    
    @validator('current_year')
    def validate_current_year(cls, v, values):
        if v < values.get('start_year', 0):
            raise ValueError("current_year must be greater than or equal to start_year")
        return v

class ESPNApiConfig(BaseModel):
    base_url: str
    request_timeout: int = 30
    rate_limit: RateLimitConfig
    endpoints: APIEndpoints
    seasons: SeasonConfig

class DuckDBStorageConfig(BaseModel):
    parquet_directory: str
    use_compression: bool = True
    compression_method: str = "zstd"
    
    @validator('parquet_directory')
    def create_directory_if_not_exists(cls, v):
        os.makedirs(v, exist_ok=True)
        return v

class DuckDBPerformanceConfig(BaseModel):
    memory_limit: str = "4GB"
    threads: int = 4
    
    @validator('threads')
    def validate_threads(cls, v):
        import multiprocessing
        max_threads = multiprocessing.cpu_count()
        if v > max_threads:
            return max_threads
        return v

class DuckDBAdvancedConfig(BaseModel):
    extension_directory: str = "extensions/"
    load_extensions: List[str] = ["json", "parquet"]

class DuckDBConfig(BaseModel):
    database_path: str
    storage: DuckDBStorageConfig
    performance: DuckDBPerformanceConfig
    advanced: DuckDBAdvancedConfig

class FeatureGroup(BaseModel):
    enabled: bool = True
    lookback_periods: Optional[List[int]] = None
    max_lookback_years: Optional[int] = None

class FeatureComputationConfig(BaseModel):
    window_sizes: List[int]
    recalculation_frequency: Literal["hourly", "daily", "weekly"] = "daily"

class FeatureStorageConfig(BaseModel):
    directory: str
    format: Literal["parquet", "csv", "feather"] = "parquet"
    
    @validator('directory')
    def create_directory_if_not_exists(cls, v):
        os.makedirs(v, exist_ok=True)
        return v

class FeatureGroupsConfig(BaseModel):
    team_stats: FeatureGroup
    player_stats: FeatureGroup
    game_context: FeatureGroup
    historical_matchups: FeatureGroup

class FeatureConfig(BaseModel):
    storage: FeatureStorageConfig
    computation: FeatureComputationConfig
    groups: FeatureGroupsConfig

class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    model_registry: str

class TrainingConfig(BaseModel):
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cv_folds: int = 5
    
    @validator('test_size', 'validation_size')
    def validate_split_sizes(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("Split sizes must be between 0 and 1")
        return v

class PyTorchModelConfig(BaseModel):
    architecture: Literal["mlp", "lstm", "transformer"] = "lstm"
    hidden_layers: List[int]
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 5

class LightGBMModelConfig(BaseModel):
    enabled: bool = True
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 100
    colsample_bytree: float = 0.8
    subsample: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

class ModelParamsConfig(BaseModel):
    pytorch: PyTorchModelConfig
    lightgbm: LightGBMModelConfig

class ModelConfig(BaseModel):
    mlflow: MLflowConfig
    training: TrainingConfig
    model_params: ModelParamsConfig

# Main configuration class that combines all configurations
class ProjectConfig(BaseModel):
    espn_api: ESPNApiConfig
    duckdb: DuckDBConfig
    features: FeatureConfig
    models: ModelConfig

# Helper functions to load config from YAML files
import yaml
from pathlib import Path

def load_config(config_dir: str = "config") -> ProjectConfig:
    """Load and validate all configuration files into a single ProjectConfig object."""
    base_path = Path(config_dir)
    
    # Load individual config files
    with open(base_path / "api_config.yaml", "r") as f:
        api_config = yaml.safe_load(f)
        
    with open(base_path / "db_config.yaml", "r") as f:
        db_config = yaml.safe_load(f)
        
    with open(base_path / "feature_config.yaml", "r") as f:
        feature_config = yaml.safe_load(f)
        
    with open(base_path / "model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    # Combine all configs into a single dictionary
    config_dict = {
        "espn_api": api_config["espn_api"],
        "duckdb": db_config["duckdb"],
        "features": feature_config["features"],
        "models": model_config["models"]
    }
    
    # Validate with Pydantic
    return ProjectConfig(**config_dict)
```

## Usage Example

```python
from src.config.settings import load_config

# Load and validate all configuration
config = load_config()

# Access specific configuration sections
api_config = config.espn_api
db_config = config.duckdb
feature_config = config.features
model_config = config.models

# Use in application
base_url = api_config.base_url
database_path = db_config.database_path
window_sizes = feature_config.computation.window_sizes
```

## Benefits of This Approach

1. **Type Safety**: Pydantic validates all configuration values and provides helpful error messages.
2. **Default Values**: Sensible defaults can be specified in the Pydantic models.
3. **Documentation**: The models themselves document the expected configuration structure.
4. **Validation Logic**: Custom validators can enforce complex constraints and relationships.
5. **IDE Support**: IDEs can provide autocomplete and type hints when using the configuration.
