# Feature Engineering Framework

## Overview

This document outlines the feature engineering framework for the NCAA Basketball Analytics project. The framework provides a structured approach to creating, computing, versioning, and managing features for predictive modeling of basketball games.

## Core Concepts

The feature engineering framework is built around several key concepts:

1. **Feature Hierarchy**: Features are organized in a hierarchical structure (base features, derived features, composite features)
2. **Feature Registry**: A central catalog of all available features
3. **Versioning**: Feature definitions and implementations are versioned for reproducibility
4. **Incremental Computation**: Efficient recalculation of only what's needed
5. **Lineage Tracking**: Clear tracking of feature dependencies and sources

![Feature Engineering Framework](https://i.imgur.com/placeholder.png)

## Feature Class Hierarchy

### Base Feature Class

All features extend the `BaseFeature` class, which provides common functionality:

```python
# src/features/base.py
from abc import ABC, abstractmethod
import pandas as pd
import polars as pl
import duckdb
import hashlib
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from src.config.logging import get_logger

class BaseFeature(ABC):
    """Base class for all features"""
    
    def __init__(
        self,
        name: str,
        version: int = 1,
        description: str = "",
        dependencies: List[str] = None,
        category: str = "uncategorized",
        tags: List[str] = None
    ):
        """
        Initialize a feature
        
        Args:
            name: Unique feature name
            version: Feature version number
            description: Human-readable description
            dependencies: Names of features this feature depends on
            category: Feature category for grouping
            tags: Additional tags for filtering
        """
        self.name = name
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self.category = category
        self.tags = tags or []
        self.id = f"{self.name}_v{self.version}"
        
        # Set up logger
        self.logger = get_logger(f"features.{self.category}.{self.name}")
    
    @property
    def signature(self) -> str:
        """Return a unique signature for this feature definition"""
        feature_def = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": sorted(self.dependencies),
            "category": self.category,
            "tags": sorted(self.tags),
            "implementation": self.__class__.__name__
        }
        return hashlib.md5(json.dumps(feature_def, sort_keys=True).encode()).hexdigest()
    
    def compute(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        db_conn: Optional[duckdb.DuckDBPyConnection] = None,
        batch_id: Optional[str] = None,
        incremental: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Compute feature values
        
        Args:
            data: Input data
            db_conn: DuckDB connection (if needed)
            batch_id: Unique identifier for this computation batch
            incremental: Whether to do incremental computation
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with feature values
        """
        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = str(uuid.uuid4())
            
        # Convert to Polars if pandas
        using_pandas = isinstance(data, pd.DataFrame)
        if using_pandas:
            data = pl.from_pandas(data)
            
        # Log computation start
        start_time = datetime.now()
        self.logger.info(
            f"Starting computation of feature {self.name} (v{self.version})",
            extra={
                "feature_id": self.id,
                "batch_id": batch_id,
                "signature": self.signature,
                "input_rows": data.shape[0],
                "incremental": incremental
            }
        )
        
        try:
            # Call implementation
            result = self._compute_impl(
                data=data, 
                db_conn=db_conn, 
                batch_id=batch_id,
                incremental=incremental,
                **kwargs
            )
            
            # Log success
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self.logger.info(
                f"Completed computation of feature {self.name} (v{self.version})",
                extra={
                    "feature_id": self.id,
                    "batch_id": batch_id,
                    "signature": self.signature,
                    "duration_ms": duration_ms,
                    "output_rows": result.shape[0],
                    "output_columns": len(result.columns)
                }
            )
            
            # Convert back to pandas if input was pandas
            if using_pandas:
                result = result.to_pandas()
                
            return result
            
        except Exception as e:
            # Log failure
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self.logger.error(
                f"Failed computation of feature {self.name} (v{self.version}): {str(e)}",
                extra={
                    "feature_id": self.id,
                    "batch_id": batch_id,
                    "signature": self.signature,
                    "duration_ms": duration_ms,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise
    
    @abstractmethod
    def _compute_impl(
        self,
        data: pl.DataFrame,
        db_conn: Optional[duckdb.DuckDBPyConnection],
        batch_id: str,
        incremental: bool,
        **kwargs
    ) -> pl.DataFrame:
        """
        Implementation of feature computation
        
        This method must be implemented by subclasses.
        """
        pass
    
    def validate(self, data: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        """
        Validate feature values
        
        Args:
            data: DataFrame containing feature values
            
        Returns:
            Validation results
        """
        # Convert to Polars if pandas
        using_pandas = isinstance(data, pd.DataFrame)
        if using_pandas:
            data = pl.from_pandas(data)
            
        # Check if feature column exists
        feature_col = self.get_output_column_name()
        if feature_col not in data.columns:
            return {
                "valid": False,
                "error": f"Feature column '{feature_col}' not found in data"
            }
            
        # Get feature values
        feature_values = data[feature_col]
        
        # Basic validation statistics
        results = {
            "valid": True,
            "count": len(feature_values),
            "null_count": feature_values.null_count(),
            "null_percentage": feature_values.null_count() / len(feature_values) * 100 if len(feature_values) > 0 else 0
        }
        
        # Add numeric statistics if applicable
        if feature_values.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            non_null_values = feature_values.drop_nulls()
            if len(non_null_values) > 0:
                results.update({
                    "min": float(non_null_values.min()),
                    "max": float(non_null_values.max()),
                    "mean": float(non_null_values.mean()),
                    "median": float(non_null_values.median()),
                    "std": float(non_null_values.std())
                })
        
        # Custom validation (implemented by subclasses)
        custom_results = self._validate_impl(data)
        if custom_results:
            results.update(custom_results)
            
        return results
    
    def _validate_impl(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Custom validation implementation
        
        This method can be overridden by subclasses for custom validation logic.
        """
        return {}
    
    def get_output_column_name(self) -> str:
        """Get the column name for this feature in the output DataFrame"""
        return self.name
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get feature metadata"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": self.dependencies,
            "category": self.category,
            "tags": self.tags,
            "signature": self.signature,
            "class": self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        return f"Feature(id={self.id}, signature={self.signature})"
```

### Feature Types

Different feature types extend the base class:

#### Statistic Features

```python
# src/features/statistic.py
from src.features.base import BaseFeature
import polars as pl
from typing import Optional, List, Dict, Any
import duckdb

class StatisticFeature(BaseFeature):
    """Base class for features based on game statistics"""
    
    def __init__(
        self,
        name: str,
        stat_column: str,
        aggregation: str = "mean",
        **kwargs
    ):
        """
        Initialize a statistic feature
        
        Args:
            name: Feature name
            stat_column: Column name in stats table
            aggregation: Aggregation function (mean, sum, max, etc.)
        """
        super().__init__(name=name, category="statistic", **kwargs)
        self.stat_column = stat_column
        self.aggregation = aggregation
    
    def _compute_impl(
        self,
        data: pl.DataFrame,
        db_conn: Optional[duckdb.DuckDBPyConnection],
        batch_id: str,
        incremental: bool,
        **kwargs
    ) -> pl.DataFrame:
        """Compute statistic feature"""
        if db_conn is None:
            raise ValueError("DuckDB connection is required for StatisticFeature")
            
        # Build SQL query based on the stats column and aggregation
        query = f"""
        SELECT
            t.team_id,
            CAST('{self.name}' AS VARCHAR) AS feature_name,
            CAST('{self.version}' AS INTEGER) AS feature_version,
            CAST('{self.signature}' AS VARCHAR) AS feature_signature,
            CAST(CURRENT_TIMESTAMP AS TIMESTAMP) AS computed_at,
            CAST('{batch_id}' AS VARCHAR) AS batch_id,
            {self.aggregation}(s.{self.stat_column}) AS {self.name}
        FROM
            dim_teams t
        JOIN
            fact_team_game_stats s ON t.team_id = s.team_id
        GROUP BY
            t.team_id
        """
        
        # Execute query
        result = db_conn.execute(query).arrow()
        
        # Convert to Polars
        return pl.from_arrow(result)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get feature metadata"""
        metadata = super().get_metadata()
        metadata.update({
            "stat_column": self.stat_column,
            "aggregation": self.aggregation
        })
        return metadata
```

#### Rolling Window Features

```python
# src/features/rolling.py
from src.features.base import BaseFeature
import polars as pl
from typing import Optional, List, Dict, Any, Union
import duckdb

class RollingWindowFeature(BaseFeature):
    """Feature based on rolling window calculations"""
    
    def __init__(
        self,
        name: str,
        base_column: str,
        window_size: int,
        aggregation: str = "mean",
        min_periods: int = 1,
        **kwargs
    ):
        """
        Initialize a rolling window feature
        
        Args:
            name: Feature name
            base_column: Column to apply window function to
            window_size: Size of rolling window
            aggregation: Aggregation function (mean, sum, max, etc.)
            min_periods: Minimum periods required for calculation
        """
        # Add window size to name if not already included
        if f"_{window_size}" not in name:
            name = f"{name}_{window_size}"
            
        super().__init__(name=name, category="rolling", **kwargs)
        self.base_column = base_column
        self.window_size = window_size
        self.aggregation = aggregation
        self.min_periods = min_periods
    
    def _compute_impl(
        self,
        data: pl.DataFrame,
        db_conn: Optional[duckdb.DuckDBPyConnection],
        batch_id: str,
        incremental: bool,
        **kwargs
    ) -> pl.DataFrame:
        """Compute rolling window feature"""
        # Ensure required columns exist
        required_columns = ["entity_id", "date", self.base_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort data by entity and date
        sorted_data = data.sort(by=["entity_id", "date"])
        
        # Compute rolling window
        result = sorted_data.groupby("entity_id").agg([
            pl.col("date"),
            pl.col(self.base_column).rolling_window(
                window_size=self.window_size,
                min_periods=self.min_periods,
                closed="left"
            ).agg_list().alias("window_values")
        ])
        
        # Apply aggregation function
        if self.aggregation == "mean":
            result = result.with_columns([
                pl.col("window_values").apply(
                    lambda x: sum(x) / len(x) if x and len(x) > 0 else None
                ).alias(self.name)
            ])
        elif self.aggregation == "sum":
            result = result.with_columns([
                pl.col("window_values").apply(
                    lambda x: sum(x) if x else None
                ).alias(self.name)
            ])
        elif self.aggregation == "max":
            result = result.with_columns([
                pl.col("window_values").apply(
                    lambda x: max(x) if x and len(x) > 0 else None
                ).alias(self.name)
            ])
        elif self.aggregation == "min":
            result = result.with_columns([
                pl.col("window_values").apply(
                    lambda x: min(x) if x and len(x) > 0 else None
                ).alias(self.name)
            ])
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
            
        # Drop intermediate columns
        result = result.drop("window_values")
        
        # Add metadata columns
        result = result.with_columns([
            pl.lit(self.name).alias("feature_name"),
            pl.lit(self.version).alias("feature_version"),
            pl.lit(self.signature).alias("feature_signature"),
            pl.lit(datetime.now()).alias("computed_at"),
            pl.lit(batch_id).alias("batch_id")
        ])
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get feature metadata"""
        metadata = super().get_metadata()
        metadata.update({
            "base_column": self.base_column,
            "window_size": self.window_size,
            "aggregation": self.aggregation,
            "min_periods": self.min_periods
        })
        return metadata
```

#### Derived Features

```python
# src/features/derived.py
from src.features.base import BaseFeature
import polars as pl
from typing import Optional, List, Dict, Any, Callable, Union
import duckdb
from datetime import datetime

class DerivedFeature(BaseFeature):
    """Feature derived from other features"""
    
    def __init__(
        self,
        name: str,
        source_features: List[str],
        transform_fn: Callable,
        **kwargs
    ):
        """
        Initialize a derived feature
        
        Args:
            name: Feature name
            source_features: List of source feature names
            transform_fn: Function to transform source features
        """
        # Set dependencies to source features
        kwargs["dependencies"] = source_features
        
        super().__init__(name=name, category="derived", **kwargs)
        self.source_features = source_features
        self.transform_fn = transform_fn
    
    def _compute_impl(
        self,
        data: pl.DataFrame,
        db_conn: Optional[duckdb.DuckDBPyConnection],
        batch_id: str,
        incremental: bool,
        **kwargs
    ) -> pl.DataFrame:
        """Compute derived feature"""
        # Ensure source features exist
        missing_features = [f for f in self.source_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing source features: {missing_features}")
        
        # Apply transform function
        result = data.with_columns([
            pl.struct(self.source_features).apply(
                lambda row: self.transform_fn(**{f: row[f] for f in self.source_features})
            ).alias(self.name)
        ])
        
        # Add metadata columns
        result = result.with_columns([
            pl.lit(self.name).alias("feature_name"),
            pl.lit(self.version).alias("feature_version"),
            pl.lit(self.signature).alias("feature_signature"),
            pl.lit(datetime.now()).alias("computed_at"),
            pl.lit(batch_id).alias("batch_id")
        ])
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get feature metadata"""
        metadata = super().get_metadata()
        metadata.update({
            "source_features": self.source_features,
            "transform_function": self.transform_fn.__name__
        })
        return metadata
```

### Domain-Specific Features

The project includes specialized basketball feature classes:

#### Team Performance Features

```python
# src/features/team_performance.py
from src.features.rolling import RollingWindowFeature
import polars as pl
from typing import Optional, List, Dict, Any
import duckdb

class TeamOffensiveRatingFeature(RollingWindowFeature):
    """Team offensive rating over a rolling window"""
    
    def __init__(self, window_size: int = 5, **kwargs):
        super().__init__(
            name=f"team_offensive_rating",
            base_column="offensive_rating",
            window_size=window_size,
            aggregation="mean",
            min_periods=1,
            category="team_performance",
            description=f"Average team offensive rating over the last {window_size} games",
            tags=["offense", "efficiency", "rating"],
            **kwargs
        )
```

#### Matchup Features

```python
# src/features/matchup.py
from src.features.base import BaseFeature
import polars as pl
from typing import Optional, List, Dict, Any
import duckdb
from datetime import datetime

class HistoricalMatchupFeature(BaseFeature):
    """Feature based on historical matchups between teams"""
    
    def __init__(
        self,
        name: str = "historical_matchup_win_pct",
        lookback_years: int = 3,
        min_games: int = 1,
        **kwargs
    ):
        super().__init__(
            name=name,
            category="matchup",
            description=f"Win percentage against opponent in the last {lookback_years} years",
            tags=["matchup", "historical", "win_percentage"],
            **kwargs
        )
        self.lookback_years = lookback_years
        self.min_games = min_games
    
    def _compute_impl(
        self,
        data: pl.DataFrame,
        db_conn: Optional[duckdb.DuckDBPyConnection],
        batch_id: str,
        incremental: bool,
        **kwargs
    ) -> pl.DataFrame:
        """Compute historical matchup feature"""
        if db_conn is None:
            raise ValueError("DuckDB connection is required for HistoricalMatchupFeature")
            
        # Build SQL query for historical matchups
        query = f"""
        WITH matchups AS (
            -- Get all matchups between teams
            SELECT
                g.game_id,
                g.game_date,
                g.season_id,
                g.home_team_id AS team_id,
                g.away_team_id AS opponent_id,
                CASE 
                    WHEN g.home_score > g.away_score THEN 1
                    ELSE 0
                END AS is_win
            FROM
                fact_games g
            WHERE
                g.home_score IS NOT NULL
                AND g.season_id >= (SELECT MAX(season_id) FROM dim_seasons) - {self.lookback_years}
                
            UNION ALL
            
            SELECT
                g.game_id,
                g.game_date,
                g.season_id,
                g.away_team_id AS team_id,
                g.home_team_id AS opponent_id,
                CASE 
                    WHEN g.away_score > g.home_score THEN 1
                    ELSE 0
                END AS is_win
            FROM
                fact_games g
            WHERE
                g.home_score IS NOT NULL
                AND g.season_id >= (SELECT MAX(season_id) FROM dim_seasons) - {self.lookback_years}
        ),
        
        matchup_stats AS (
            -- Calculate stats for each team-opponent pair
            SELECT
                team_id,
                opponent_id,
                COUNT(*) AS games_played,
                SUM(is_win) AS wins,
                CAST(SUM(is_win) AS FLOAT) / COUNT(*) AS win_percentage
            FROM
                matchups
            GROUP BY
                team_id, opponent_id
            HAVING
                COUNT(*) >= {self.min_games}
        )
        
        -- Final output
        SELECT
            ms.team_id,
            ms.opponent_id,
            CAST('{self.name}' AS VARCHAR) AS feature_name,
            CAST({self.version} AS INTEGER) AS feature_version,
            CAST('{self.signature}' AS VARCHAR) AS feature_signature,
            CAST(CURRENT_TIMESTAMP AS TIMESTAMP) AS computed_at,
            CAST('{batch_id}' AS VARCHAR) AS batch_id,
            ms.games_played,
            ms.wins,
            ms.win_percentage AS {self.name}
        FROM
            matchup_stats ms
        """
        
        # Execute query
        result = db_conn.execute(query).arrow()
        
        # Convert to Polars
        return pl.from_arrow(result)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get feature metadata"""
        metadata = super().get_metadata()
        metadata.update({
            "lookback_years": self.lookback_years,
            "min_games": self.min_games
        })
        return metadata
```

## Feature Registry

The Feature Registry manages feature definitions, versioning, and dependencies:

```python
# src/features/registry.py
from typing import Dict, List, Any, Type, Optional, Set, Union
import pandas as pd
import polars as pl
import duckdb
import uuid
import json
from datetime import datetime
import os
from pathlib import Path
import importlib
import inspect

from src.features.base import BaseFeature
from src.config.logging import get_logger

logger = get_logger("features.registry")

class FeatureRegistry:
    """Registry for feature definitions and implementations"""
    
    def __init__(
        self,
        storage_dir: str,
        db_path: Optional[str] = None,
        auto_discover: bool = True
    ):
        """
        Initialize feature registry
        
        Args:
            storage_dir: Directory for feature storage
            db_path: Path to DuckDB database
            auto_discover: Whether to auto-discover features
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.db_conn = None
        if db_path:
            self.db_conn = duckdb.connect(db_path)
            
        self.registry = {}  # name -> List[Feature]
        self.instances = {}  # id -> Feature instance
        
        self.logger = logger
        
        if auto_discover:
            self.discover_features()
    
    def discover_features(self, module_prefix: str = "src.features"):
        """
        Discover feature implementations via module inspection
        
        Args:
            module_prefix: Package prefix for feature modules
        """
        self.logger.info(f"Discovering features with prefix: {module_prefix}")
        
        # Import base modules
        try:
            base_module = importlib.import_module(module_prefix)
        except ImportError:
            self.logger.warning(f"Could not import module: {module_prefix}")
            return
            
        # Get feature modules
        feature_modules = []
        for module_name in dir(base_module):
            if module_name.startswith('_'):
                continue
                
            try:
                module = importlib.import_module(f"{module_prefix}.{module_name}")
                feature_modules.append(module)
                self.logger.debug(f"Imported module: {module_prefix}.{module_name}")
            except ImportError:
                pass
        
        # Find feature classes
        feature_classes = []
        for module in feature_modules:
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseFeature) and obj != BaseFeature:
                    feature_classes.append(obj)
                    self.logger.debug(f"Found feature class: {obj.__name__}")
        
        # Register feature classes
        for feature_class in feature_classes:
            try:
                # Check if class has a default constructor
                if '__init__' in feature_class.__dict__:
                    sig = inspect.signature(feature_class.__init__)
                    if all(param.default != inspect.Parameter.empty 
                           for param in list(sig.parameters.values())[1:]):
                        # Create instance and register
                        feature = feature_class()
                        self.register_feature(feature)
                        self.logger.info(f"Auto-registered feature: {feature.id}")
            except Exception as e:
                self.logger.warning(f"Could not auto-register feature class {feature_class.__name__}: {str(e)}")
    
    def register_feature(self, feature: BaseFeature) -> str:
        """
        Register a feature
        
        Args:
            feature: Feature instance
            
        Returns:
            Feature ID
        """
        feature_id = feature.id
        feature_name = feature.name
        
        # Add to registry by name
        if feature_name not in self.registry:
            self.registry[feature_name] = []
            
        # Check if feature version already exists
        existing_versions = [f.version for f in self.registry[feature_name]]
        if feature.version in existing_versions:
            raise ValueError(f"Feature {feature_name} v{feature.version} already registered")
            
        # Add to registry
        self.registry[feature_name].append(feature)
        self.instances[feature_id] = feature
        
        self.logger.info(f"Registered feature: {feature_id}", extra={"signature": feature.signature})
        
        return feature_id
    
    def get_feature(self, name: str, version: Optional[int] = None) -> BaseFeature:
        """
        Get a feature by name and optional version
        
        Args:
            name: Feature name
            version: Feature version (if None, get latest)
            
        Returns:
            Feature instance
        """
        if name not in self.registry:
            raise KeyError(f"Feature {name} not registered")
            
        features = self.registry[name]
        
        if version is not None:
            # Get specific version
            for feature in features:
                if feature.version == version:
                    return feature
            raise KeyError(f"Feature {name} v{version} not found")
        else:
            # Get latest version
            return max(features, key=lambda f: f.version)
    
    def list_features(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available features
        
        Args:
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            List of feature metadata
        """
        features = list(self.instances.values())
        
        # Apply filters
        if category:
            features = [f for f in features if f.category == category]
            
        if tags:
            features = [f for f in features if all(tag in f.tags for tag in tags)]
            
        # Get metadata
        return [f.get_metadata() for f in features]
    
    def compute_feature(
        self,
        name: str,
        data: Union[pd.DataFrame, pl.DataFrame],
        version: Optional[int] = None,
        batch_id: Optional[str] = None,
        incremental: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Compute a feature
        
        Args:
            name: Feature name
            data: Input data
            version: Feature version (if None, use latest)
            batch_id: Batch ID for this computation
            incremental: Whether to do incremental computation
            **kwargs: Additional parameters for computation
            
        Returns:
            DataFrame with computed feature
        """
        feature = self.get_feature(name, version)
        
        self.logger.info(
            f"Computing feature: {feature.id}",
            extra={
                "batch_id": batch_id or str(uuid.uuid4()),
                "incremental": incremental,
                "input_rows": len(data)
            }
        )
        
        return feature.compute(
            data=data,
            db_conn=self.db_conn,
            batch_id=batch_id or str(uuid.uuid4()),
            incremental=incremental,
            **kwargs
        )
    
    def compute_feature_group(
        self,
        category: str,
        data: Union[pd.DataFrame, pl.DataFrame],
        batch_id: Optional[str] = None,
        incremental: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Compute all features in a category
        
        Args:
            category: Feature category
            data: Input data
            batch_id: Batch ID for this computation
            incremental: Whether to do incremental computation
            **kwargs: Additional parameters for computation
            
        Returns:
            DataFrame with computed features
        """
        # Get features in category
        features_meta = self.list_features(category=category)
        
        if not features_meta:
            raise ValueError(f"No features found in category: {category}")
            
        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = str(uuid.uuid4())
            
        self.logger.info(
            f"Computing feature group: {category}",
            extra={
                "batch_id": batch_id,
                "incremental": incremental,
                "feature_count": len(features_meta),
                "input_rows": len(data)
            }
        )
        
        # Initialize result with input data
        result = data
        
        # Compute each feature
        for meta in features_meta:
            feature_id = meta["id"]
            feature = self.instances[feature_id]
            
            # Compute feature
            feature_result = feature.compute(
                data=result,
                db_conn=self.db_conn,
                batch_id=batch_id,
                incremental=incremental,
                **kwargs
            )
            
            # Add feature column to result if not already present
            feature_col = feature.get_output_column_name()
            if feature_col not in result.columns:
                if isinstance(result, pd.DataFrame):
                    result[feature_col] = feature_result[feature_col]
                else:  # polars DataFrame
                    result = result.with_columns([
                        feature_result.select(feature_col).rename({feature_col: feature_col})
                    ])
        
        return result
    
    def validate_features(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate computed features
        
        Args:
            data: DataFrame with computed features
            feature_names: List of feature names to validate (if None, validate all)
            
        Returns:
            Validation results by feature
        """
        if feature_names is None:
            # Get feature names from data columns
            feature_names = []
            for name in self.registry:
                feature = self.get_feature(name)
                col_name = feature.get_output_column_name()
                if col_name in data.columns:
                    feature_names.append(name)
        
        self.logger.info(
            f"Validating {len(feature_names)} features",
            extra={"feature_names": feature_names}
        )
        
        # Validate each feature
        results = {}
        for name in feature_names:
            feature = self.get_feature(name)
            feature_results = feature.validate(data)
            results[name] = feature_results
            
            # Log validation results
            if feature_results.get("valid", False):
                self.logger.info(
                    f"Feature validation passed: {feature.id}",
                    extra={"validation_results": feature_results}
                )
            else:
                self.logger.warning(
                    f"Feature validation failed: {feature.id}",
                    extra={"validation_results": feature_results}
                )
        
        return results
    
    def save_feature_metadata(self, output_path: Optional[str] = None) -> str:
        """
        Save feature registry metadata to JSON
        
        Args:
            output_path: Path to save metadata (if None, use default)
            
        Returns:
            Path to saved metadata file
        """
        if output_path is None:
            output_path = self.storage_dir / "feature_registry.json"
            
        metadata = {
            "features": self.list_features(),
            "updated_at": datetime.now().isoformat()
        }
        
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Saved feature registry metadata to: {output_path}")
        
        return str(output_path)
    
    def close(self):
        """Close connections"""
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
```

## Feature Manager

The Feature Manager orchestrates feature computation workflows:

```python
# src/features/manager.py
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import pandas as pd
import polars as pl
import duckdb
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
import networkx as nx

from src.features.registry import FeatureRegistry
from src.features.base import BaseFeature
from src.config.logging import get_logger

logger = get_logger("features.manager")

class FeatureManager:
    """
    Orchestrates feature computation and management
    """
    
    def __init__(
        self,
        registry: FeatureRegistry,
        output_dir: str,
        db_path: Optional[str] = None
    ):
        """
        Initialize feature manager
        
        Args:
            registry: Feature registry
            output_dir: Directory for output files
            db_path: Path to DuckDB database
        """
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.db_conn = None
        if db_path:
            self.db_conn = duckdb.connect(db_path)
            
        self.logger = logger
    
    def _get_dependency_order(self, feature_names: List[str]) -> List[str]:
        """
        Determine the order to compute features based on dependencies
        
        Args:
            feature_names: List of feature names to compute
            
        Returns:
            Ordered list of feature names
        """
        # Build dependency graph
        G = nx.DiGraph()
        
        # Add nodes
        for name in feature_names:
            G.add_node(name)
            
        # Add edges for dependencies
        for name in feature_names:
            feature = self.registry.get_feature(name)
            for dep in feature.dependencies:
                if dep in feature_names:
                    G.add_edge(dep, name)  # dep -> name
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            raise ValueError(f"Cyclic dependencies detected: {cycles}")
            
        # Get topological sort
        return list(nx.topological_sort(G))
    
    def compute_features(
        self,
        feature_names: List[str],
        input_data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
        input_query: Optional[str] = None,
        output_file: Optional[str] = None,
        batch_id: Optional[str] = None,
        incremental: bool = True,
        validate: bool = True
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Compute a set of features
        
        Args:
            feature_names: List of feature names to compute
            input_data: Input DataFrame (if not provided, use input_query)
            input_query: SQL query to get input data (if not provided, use input_data)
            output_file: Path to save results (if None, don't save)
            batch_id: Batch ID for this computation
            incremental: Whether to do incremental computation
            validate: Whether to validate results
            
        Returns:
            DataFrame with computed features
        """
        if input_data is None and input_query is None:
            raise ValueError("Either input_data or input_query must be provided")
            
        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = str(uuid.uuid4())
            
        self.logger.info(
            f"Computing {len(feature_names)} features",
            extra={
                "batch_id": batch_id,
                "feature_names": feature_names,
                "incremental": incremental,
                "validate": validate
            }
        )
        
        # Get input data
        if input_data is None:
            if self.db_conn is None:
                raise ValueError("DuckDB connection required for input_query")
                
            self.logger.info(f"Executing input query", extra={"batch_id": batch_id})
            input_data = pl.from_arrow(self.db_conn.execute(input_query).arrow())
            
        # Convert to polars if pandas
        is_pandas = isinstance(input_data, pd.DataFrame)
        if is_pandas:
            input_data = pl.from_pandas(input_data)
            
        # Get computation order
        ordered_features = self._get_dependency_order(feature_names)
        
        self.logger.info(
            f"Computation order determined",
            extra={
                "batch_id": batch_id,
                "ordered_features": ordered_features
            }
        )
        
        # Initialize result with input data
        result = input_data
        
        # Compute each feature
        for name in ordered_features:
            feature = self.registry.get_feature(name)
            
            self.logger.info(
                f"Computing feature: {feature.id}",
                extra={"batch_id": batch_id}
            )
            
            # Check if dependencies are in result
            missing_deps = [dep for dep in feature.dependencies if dep not in result.columns]
            if missing_deps:
                raise ValueError(f"Missing dependencies for {feature.id}: {missing_deps}")
                
            # Compute feature
            feature_result = feature.compute(
                data=result,
                db_conn=self.db_conn,
                batch_id=batch_id,
                incremental=incremental
            )
            
            # Add feature column to result
            feature_col = feature.get_output_column_name()
            result = result.with_columns([
                feature_result.select(feature_col).rename({feature_col: feature_col})
            ])
        
        # Validate if requested
        if validate:
            validation_results = self.registry.validate_features(result, feature_names)
            
            # Check for validation failures
            failures = [name for name, res in validation_results.items() if not res.get("valid", False)]
            if failures:
                self.logger.warning(
                    f"Validation failed for {len(failures)} features",
                    extra={
                        "batch_id": batch_id,
                        "failed_features": failures
                    }
                )
        
        # Save if output file specified
        if output_file:
            output_path = self.output_dir / output_file
            if output_path.suffix == ".parquet":
                result.write_parquet(output_path)
            elif output_path.suffix == ".csv":
                result.write_csv(output_path)
            else:
                result.write_parquet(f"{output_path}.parquet")
                
            self.logger.info(
                f"Saved results to: {output_path}",
                extra={
                    "batch_id": batch_id,
                    "output_rows": result.shape[0],
                    "output_columns": result.shape[1]
                }
            )
        
        # Convert back to pandas if input was pandas
        if is_pandas:
            result = result.to_pandas()
            
        return result
    
    def compute_feature_groups(
        self,
        categories: List[str],
        input_data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
        input_query: Optional[str] = None,
        output_file: Optional[str] = None,
        batch_id: Optional[str] = None,
        incremental: bool = True,
        validate: bool = True
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Compute all features in specified categories
        
        Args:
            categories: List of categories to compute
            input_data: Input DataFrame (if not provided, use input_query)
            input_query: SQL query to get input data (if not provided, use input_data)
            output_file: Path to save results (if None, don't save)
            batch_id: Batch ID for this computation
            incremental: Whether to do incremental computation
            validate: Whether to validate results
            
        Returns:
            DataFrame with computed features
        """
        # Get all feature names in categories
        feature_names = []
        for category in categories:
            features_meta = self.registry.list_features(category=category)
            for meta in features_meta:
                feature_names.append(meta["name"])
                
        # Deduplicate
        feature_names = list(set(feature_names))
        
        self.logger.info(
            f"Computing features for {len(categories)} categories",
            extra={
                "categories": categories,
                "feature_count": len(feature_names)
            }
        )
        
        # Compute features
        return self.compute_features(
            feature_names=feature_names,
            input_data=input_data,
            input_query=input_query,
            output_file=output_file,
            batch_id=batch_id,
            incremental=incremental,
            validate=validate
        )
    
    def load_features(
        self,
        feature_names: List[str],
        entity_ids: Optional[List[str]] = None,
        latest_only: bool = True
    ) -> pl.DataFrame:
        """
        Load computed features from storage
        
        Args:
            feature_names: List of feature names to load
            entity_ids: Optional list of entity IDs to filter by
            latest_only: Whether to get only the latest version
            
        Returns:
            DataFrame with loaded features
        """
        if self.db_conn is None:
            raise ValueError("DuckDB connection required for load_features")
            
        # Build query
        feature_list = ", ".join([f"'{name}'" for name in feature_names])
        
        query = f"""
        WITH latest_versions AS (
            SELECT
                feature_name,
                MAX(feature_version) AS latest_version
            FROM
                features
            WHERE
                feature_name IN ({feature_list})
            GROUP BY
                feature_name
        )
        
        SELECT
            f.*
        FROM
            features f
        """
        
        if latest_only:
            query += f"""
            JOIN
                latest_versions lv ON f.feature_name = lv.feature_name
                AND f.feature_version = lv.latest_version
            """
            
        if entity_ids:
            entity_list = ", ".join([f"'{id}'" for id in entity_ids])
            query += f"""
            WHERE
                f.entity_id IN ({entity_list})
            """
            
        # Execute query
        self.logger.info(
            f"Loading features",
            extra={
                "feature_names": feature_names,
                "entity_count": len(entity_ids) if entity_ids else "all",
                "latest_only": latest_only
            }
        )
        
        result = pl.from_arrow(self.db_conn.execute(query).arrow())
        
        self.logger.info(
            f"Loaded features",
            extra={
                "rows": result.shape[0],
                "columns": result.shape[1]
            }
        )
        
        return result
    
    def close(self):
        """Close connections"""
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
```

## Batch Calculation with Airflow

Airflow DAGs orchestrate feature calculation:

```python
# airflow/dags/feature_engineering/team_features_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.features.registry import FeatureRegistry
from src.features.manager import FeatureManager
from src.config.settings import load_config
from src.config.logging import get_logger

config = load_config()
logger = get_logger("airflow.dags.feature_engineering")

default_args = {
    'owner': 'ncaa_analytics',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'generate_team_features',
    default_args=default_args,
    description='Generate team-level features',
    schedule_interval='0 8 * * *',  # Daily at 8 AM
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['feature_engineering', 'teams'],
    params={
        'force_full_recalculation': Param(False, type='boolean', description='Force recalculation of all features'),
        'target_categories': Param(['team_performance', 'team_efficiency', 'team_shooting'], type='array', description='Feature categories to calculate'),
    },
) as dag:
    
    # Function to determine calculation type
    def determine_calculation_type(**context):
        # Check if force_full_recalculation is set
        if context['params'].get('force_full_recalculation', False):
            return 'full_recalculation'
            
        # Check the day of week - do full recalculation on Sundays
        if datetime.now().weekday() == 6:
            return 'full_recalculation'
            
        # Check if first run of the month - do full recalculation
        if datetime.now().day == 1:
            return 'full_recalculation'
            
        # Otherwise do incremental
        return 'incremental_calculation'
    
    branch_task = BranchPythonOperator(
        task_id='determine_calculation_type',
        python_callable=determine_calculation_type,
    )
    
    # Function to set up feature registry
    def setup_feature_registry(**context):
        registry = FeatureRegistry(
            storage_dir=config.features.storage.directory,
            db_path=config.duckdb.database_path,
            auto_discover=True
        )
        
        # Log discovered features
        features = registry.list_features()
        
        logger.info(
            f"Feature registry initialized with {len(features)} features",
            extra={
                "feature_count": len(features),
                "categories": list(set(f['category'] for f in features))
            }
        )
        
        # Save metadata
        registry.save_feature_metadata()
        
        # Return target categories
        return context['params'].get('target_categories', ['team_performance', 'team_efficiency', 'team_shooting'])
    
    setup_registry_task = PythonOperator(
        task_id='setup_feature_registry',
        python_callable=setup_feature_registry,
    )
    
    # Function to perform full recalculation
    def perform_full_recalculation(**context):
        # Get target categories
        categories = context['ti'].xcom_pull(task_ids='setup_feature_registry')
        
        # Initialize registry and manager
        registry = FeatureRegistry(
            storage_dir=config.features.storage.directory,
            db_path=config.duckdb.database_path,
            auto_discover=True
        )
        
        manager = FeatureManager(
            registry=registry,
            output_dir=config.features.storage.directory,
            db_path=config.duckdb.database_path
        )
        
        # Query for all historical team data
        query = """
        SELECT
            t.team_id AS entity_id,
            g.game_date AS date,
            g.season_id,
            ts.*
        FROM
            dim_teams t
        JOIN
            fact_team_game_stats ts ON t.team_id = ts.team_id
        JOIN
            fact_games g ON ts.game_id = g.game_id
        WHERE
            g.game_date <= CURRENT_DATE
        ORDER BY
            t.team_id, g.game_date
        """
        
        # Compute features
        batch_id = f"full_recalc_{datetime.now().strftime('%Y%m%d')}"
        
        result = manager.compute_feature_groups(
            categories=categories,
            input_query=query,
            output_file="team_features_latest.parquet",
            batch_id=batch_id,
            incremental=False,
            validate=True
        )
        
        # Also save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result.write_parquet(f"{config.features.storage.directory}/team_features_{timestamp}.parquet")
        
        # Close connections
        manager.close()
        registry.close()
        
        return {
            "calculation_type": "full",
            "batch_id": batch_id,
            "categories": categories,
            "feature_count": len(result.columns) - 3,  # Subtract entity_id, date, season_id
            "entity_count": result.select(pl.col("entity_id")).n_unique(),
            "timestamp": timestamp
        }
    
    full_recalculation_task = PythonOperator(
        task_id='full_recalculation',
        python_callable=perform_full_recalculation,
    )
    
    # Function to perform incremental calculation
    def perform_incremental_calculation(**context):
        # Get target categories
        categories = context['ti'].xcom_pull(task_ids='setup_feature_registry')
        
        # Initialize registry and manager
        registry = FeatureRegistry(
            storage_dir=config.features.storage.directory,
            db_path=config.duckdb.database_path,
            auto_discover=True
        )
        
        manager = FeatureManager(
            registry=registry,
            output_dir=config.features.storage.directory,
            db_path=config.duckdb.database_path
        )
        
        # Query for recent team data (last 30 days plus buffer)
        query = """
        SELECT
            t.team_id AS entity_id,
            g.game_date AS date,
            g.season_id,
            ts.*
        FROM
            dim_teams t
        JOIN
            fact_team_game_stats ts ON t.team_id = ts.team_id
        JOIN
            fact_games g ON ts.game_id = g.game_id
        WHERE
            g.game_date >= CURRENT_DATE - INTERVAL '45 days'
            AND g.game_date <= CURRENT_DATE
        ORDER BY
            t.team_id, g.game_date
        """
        
        # Compute features
        batch_id = f"incremental_{datetime.now().strftime('%Y%m%d')}"
        
        result = manager.compute_feature_groups(
            categories=categories,
            input_query=query,
            output_file="team_features_latest.parquet",
            batch_id=batch_id,
            incremental=True,
            validate=True
        )
        
        # Also save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result.write_parquet(f"{config.features.storage.directory}/team_features_incremental_{timestamp}.parquet")
        
        # Close connections
        manager.close()
        registry.close()
        
        return {
            "calculation_type": "incremental",
            "batch_id": batch_id,
            "categories": categories,
            "feature_count": len(result.columns) - 3,  # Subtract entity_id, date, season_id
            "entity_count": result.select(pl.col("entity_id")).n_unique(),
            "timestamp": timestamp
        }
    
    incremental_calculation_task = PythonOperator(
        task_id='incremental_calculation',
        python_callable=perform_incremental_calculation,
    )
    
    # Final task to log results
    def log_calculation_results(**context):
        # Get results from the appropriate task
        calc_type = context['ti'].xcom_pull(task_ids='determine_calculation_type')
        if calc_type == 'full_recalculation':
            results = context['ti'].xcom_pull(task_ids='full_recalculation')
        else:
            results = context['ti'].xcom_pull(task_ids='incremental_calculation')
            
        logger.info(
            f"Feature calculation complete",
            extra=results
        )
        
        return results
    
    log_results_task = PythonOperator(
        task_id='log_calculation_results',
        python_callable=log_calculation_results,
    )
    
    # Define task flow
    setup_registry_task >> branch_task
    branch_task >> [full_recalculation_task, incremental_calculation_task]
    full_recalculation_task >> log_results_task
    incremental_calculation_task >> log_results_task
```

## NCAA Basketball Features

### Team Performance Features

Key features calculated for team performance:

1. **Offensive Efficiency**
   - Points per 100 possessions
   - Shooting percentages (FG%, 3P%, FT%)
   - Turnover rate
   - Offensive rebound rate

2. **Defensive Efficiency**
   - Points allowed per 100 possessions
   - Opponent shooting percentages
   - Forced turnover rate
   - Defensive rebound rate

3. **Pace Factors**
   - Possessions per game
   - Average possession length
   - Fast break points percentage

4. **Time-Series Features**
   - Rolling averages (5-game, 10-game windows)
   - Season-to-date statistics
   - Trend indicators (improving/declining)

5. **Matchup-Specific Features**
   - Historical performance vs. specific opponents
   - Performance against similar team styles
   - Home/away/neutral court factors

### Implementation Example: Team Offensive Rating

```python
# src/features/team_features.py
from src.features.base import BaseFeature
from src.features.rolling import RollingWindowFeature
import polars as pl
import duckdb
from typing import Optional, Dict, Any
from datetime import datetime

class TeamOffensiveRatingFeature(RollingWindowFeature):
    """Team offensive rating over a rolling window"""
    
    def __init__(self, window_size: int = 5, version: int = 1):
        super().__init__(
            name=f"team_offensive_rating_{window_size}",
            base_column="offensive_rating",
            window_size=window_size,
            aggregation="mean",
            version=version,
            category="team_performance",
            description=f"Average offensive rating over the last {window_size} games",
            tags=["offense", "efficiency", "rating"]
        )

class TeamEffectiveFGPctFeature(RollingWindowFeature):
    """Team effective field goal percentage over a rolling window"""
    
    def __init__(self, window_size: int = 5, version: int = 1):
        super().__init__(
            name=f"team_effective_fg_pct_{window_size}",
            base_column="effective_fg_pct",
            window_size=window_size,
            aggregation="mean",
            version=version,
            category="team_shooting",
            description=f"Average effective field goal percentage over the last {window_size} games",
            tags=["offense", "shooting", "efficiency"]
        )

class TeamTurnoverRateFeature(RollingWindowFeature):
    """Team turnover rate over a rolling window"""
    
    def __init__(self, window_size: int = 5, version: int = 1):
        super().__init__(
            name=f"team_turnover_rate_{window_size}",
            base_column="turnover_rate",
            window_size=window_size,
            aggregation="mean",
            version=version,
            category="team_efficiency",
            description=f"Average turnover rate over the last {window_size} games",
            tags=["offense", "turnovers", "efficiency"]
        )

class TeamDefensiveRatingFeature(RollingWindowFeature):
    """Team defensive rating over a rolling window"""
    
    def __init__(self, window_size: int = 5, version: int = 1):
        super().__init__(
            name=f"team_defensive_rating_{window_size}",
            base_column="defensive_rating",
            window_size=window_size,
            aggregation="mean",
            version=version,
            category="team_performance",
            description=f"Average defensive rating over the last {window_size} games",
            tags=["defense", "efficiency", "rating"]
        )

class TeamNetRatingFeature(BaseFeature):
    """Team net rating (offensive - defensive)"""
    
    def __init__(
        self,
        window_size: int = 5,
        version: int = 1
    ):
        # Set dependencies to source features
        offrtg_name = f"team_offensive_rating_{window_size}"
        defrtg_name = f"team_defensive_rating_{window_size}"
        
        super().__init__(
            name=f"team_net_rating_{window_size}",
            version=version,
            dependencies=[offrtg_name, defrtg_name],
            category="team_performance",
            description=f"Net rating (offensive - defensive) over the last {window_size} games",
            tags=["performance", "efficiency", "rating"]
        )
        self.window_size = window_size
    
    def _compute_impl(
        self,
        data: pl.DataFrame,
        db_conn: Optional[duckdb.DuckDBPyConnection],
        batch_id: str,
        incremental: bool,
        **kwargs
    ) -> pl.DataFrame:
        """Compute net rating feature"""
        # Get source feature names
        offrtg_name = f"team_offensive_rating_{self.window_size}"
        defrtg_name = f"team_defensive_rating_{self.window_size}"
        
        # Ensure source features exist
        if offrtg_name not in data.columns or defrtg_name not in data.columns:
            missing = []
            if offrtg_name not in data.columns:
                missing.append(offrtg_name)
            if defrtg_name not in data.columns:
                missing.append(defrtg_name)
            raise ValueError(f"Missing source features: {missing}")
        
        # Calculate net rating
        output_name = self.get_output_column_name()
        result = data.with_columns([
            (pl.col(offrtg_name) - pl.col(defrtg_name)).alias(output_name)
        ])
        
        # Add metadata columns
        result = result.with_columns([
            pl.lit(self.name).alias("feature_name"),
            pl.lit(self.version).alias("feature_version"),
            pl.lit(self.signature).alias("feature_signature"),
            pl.lit(datetime.now()).alias("computed_at"),
            pl.lit(batch_id).alias("batch_id")
        ])
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get feature metadata"""
        metadata = super().get_metadata()
        metadata.update({
            "window_size": self.window_size
        })
        return metadata
```

### Example Feature Calculation Workflow

```python
# scripts/calculate_team_features.py
import argparse
import os
import sys
from datetime import datetime
import polars as pl
import duckdb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import load_config
from src.features.registry import FeatureRegistry
from src.features.manager import FeatureManager
from src.config.logging import get_logger, configure_logging

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Calculate team features")
    parser.add_argument("--incremental", action="store_true", help="Perform incremental calculation")
    parser.add_argument("--categories", nargs="+", default=["team_performance", "team_efficiency", "team_shooting"],
                      help="Feature categories to calculate")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--output-file", default="team_features_latest.parquet", help="Output file name")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_dir)
    
    # Configure logging
    logger = configure_logging()
    
    logger.info(
        f"Starting team feature calculation",
        extra={
            "incremental": args.incremental,
            "categories": args.categories,
            "output_file": args.output_file
        }
    )
    
    # Initialize feature registry
    registry = FeatureRegistry(
        storage_dir=config.features.storage.directory,
        db_path=config.duckdb.database_path,
        auto_discover=True
    )
    
    # Initialize feature manager
    manager = FeatureManager(
        registry=registry,
        output_dir=config.features.storage.directory,
        db_path=config.duckdb.database_path
    )
    
    # Determine query based on calculation type
    if args.incremental:
        # Query for recent data only
        query = """
        SELECT
            t.team_id AS entity_id,
            g.game_date AS date,
            g.season_id,
            ts.*
        FROM
            dim_teams t
        JOIN
            fact_team_game_stats ts ON t.team_id = ts.team_id
        JOIN
            fact_games g ON ts.game_id = g.game_id
        WHERE
            g.game_date >= CURRENT_DATE - INTERVAL '45 days'
            AND g.game_date <= CURRENT_DATE
        ORDER BY
            t.team_id, g.game_date
        """
    else:
        # Query for all historical data
        query = """
        SELECT
            t.team_id AS entity_id,
            g.game_date AS date,
            g.season_id,
            ts.*
        FROM
            dim_teams t
        JOIN
            fact_team_game_stats ts ON t.team_id = ts.team_id
        JOIN
            fact_games g ON ts.game_id = g.game_id
        WHERE
            g.game_date <= CURRENT_DATE
        ORDER BY
            t.team_id, g.game_date
        """
    
    # Generate batch ID
    batch_id = f"{'incr' if args.incremental else 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Compute features
    try:
        result = manager.compute_feature_groups(
            categories=args.categories,
            input_query=query,
            output_file=args.output_file,
            batch_id=batch_id,
            incremental=args.incremental,
            validate=True
        )
        
        # Log success
        logger.info(
            f"Feature calculation complete",
            extra={
                "batch_id": batch_id,
                "feature_count": len(result.columns) - 3,  # Subtract entity_id, date, season_id
                "row_count": result.shape[0],
                "entity_count": result.select("entity_id").n_unique()
            }
        )
        
        # Also save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result.write_parquet(
            f"{config.features.storage.directory}/team_features_{'incr' if args.incremental else 'full'}_{timestamp}.parquet"
        )
        
    except Exception as e:
        logger.error(
            f"Feature calculation failed: {str(e)}",
            extra={"batch_id": batch_id},
            exc_info=True
        )
        sys.exit(1)
        
    finally:
        # Close connections
        manager.close()
        registry.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Feature Storage and Versioning

Features are stored in both raw and processed formats:

1. **Parquet Files**:
   - `team_features_latest.parquet`: Latest feature values
   - `team_features_{timestamp}.parquet`: Historical feature snapshots
   - `team_features_incremental_{timestamp}.parquet`: Incremental updates

2. **Feature Registry Metadata**:
   - `feature_registry.json`: Catalog of all registered features
   - Contains versioning and lineage information

3. **DuckDB Integration**:
   - Features made available through SQL views
   - Efficient querying and joining with fact tables

## Best Practices

### 1. Feature Design Principles

1. **Single Responsibility**: Each feature computes one specific characteristic
2. **Clear Dependencies**: Dependencies are explicitly declared
3. **Versioned Definitions**: Features are versioned for reproducibility
4. **Testable**: Features can be validated independently
5. **Documented**: Features include detailed descriptions and metadata

### 2. Performance Considerations

1. **Incremental Computation**: Only recalculate what's changed
2. **Efficient Storage**: Use Parquet for column-oriented storage
3. **Lazy Evaluation**: Use Polars lazy evaluation for complex operations
4. **Parallelization**: Take advantage of parallel processing
5. **Caching**: Cache intermediate results where appropriate

### 3. Feature Engineering Guidelines

1. **Domain Knowledge**: Incorporate basketball-specific insights
2. **Feature Families**: Group related features (e.g., different window sizes)
3. **Feature Selection**: Not all features are equally valuable
4. **Feature Scaling**: Consider normalization for model input
5. **Feature Documentation**: Document meaning and interpretation

## Testing and Validation

Features include built-in validation capabilities:

```python
# src/features/testing/validators.py
from typing import Dict, Any, List, Callable, Optional
import polars as pl
import numpy as np

class FeatureValidator:
    """Validator for feature values"""
    
    @staticmethod
    def range_check(
        data: pl.DataFrame,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check if values are within a specified range
        
        Args:
            data: DataFrame containing feature values
            column: Column name to check
            min_val: Minimum allowed value (None for no minimum)
            max_val: Maximum allowed value (None for no maximum)
            
        Returns:
            Validation results
        """
        if column not in data.columns:
            return {
                "valid": False,
                "error": f"Column '{column}' not found in data"
            }
            
        # Get column statistics
        col_stats = data.select([
            pl.col(column).min().alias("min"),
            pl.col(column).max().alias("max"),
            pl.col(column).mean().alias("mean"),
            pl.col(column).std().alias("std"),
            pl.col(column).null_count().alias("null_count"),
            pl.count().alias("total_count")
        ]).row(0)
        
        # Check if values are within range
        min_value = col_stats["min"]
        max_value = col_stats["max"]
        
        results = {
            "column": column,
            "min": min_value,
            "max": max_value,
            "mean": col_stats["mean"],
            "std": col_stats["std"],
            "null_count": col_stats["null_count"],
            "total_count": col_stats["total_count"],
            "null_percentage": col_stats["null_count"] / col_stats["total_count"] * 100 if col_stats["total_count"] > 0 else 0
        }
        
        # Validate minimum value
        if min_val is not None and min_value < min_val:
            results["valid"] = False
            results["error"] = f"Minimum value {min_value} is less than required minimum {min_val}"
            return results
            
        # Validate maximum value
        if max_val is not None and max_value > max_val:
            results["valid"] = False
            results["error"] = f"Maximum value {max_value} is greater than required maximum {max_val}"
            return results
            
        # All checks passed
        results["valid"] = True
        return results
    
    @staticmethod
    def null_check(
        data: pl.DataFrame,
        column: str,
        max_null_percentage: float = 10.0
    ) -> Dict[str, Any]:
        """
        Check if null percentage is within acceptable limits
        
        Args:
            data: DataFrame containing feature values
            column: Column name to check
            max_null_percentage: Maximum allowed null percentage
            
        Returns:
            Validation results
        """
        if column not in data.columns:
            return {
                "valid": False,
                "error": f"Column '{column}' not found in data"
            }
            
        # Get null statistics
        null_stats = data.select([
            pl.col(column).null_count().alias("null_count"),
            pl.count().alias("total_count")
        ]).row(0)
        
        null_count = null_stats["null_count"]
        total_count = null_stats["total_count"]
        null_percentage = null_count / total_count * 100 if total_count > 0 else 0
        
        results = {
            "column": column,
            "null_count": null_count,
            "total_count": total_count,
            "null_percentage": null_percentage
        }
        
        # Validate null percentage
        if null_percentage > max_null_percentage:
            results["valid"] = False
            results["error"] = f"Null percentage {null_percentage:.2f}% exceeds maximum allowed {max_null_percentage}%"
            return results
            
        # All checks passed
        results["valid"] = True
        return results
    
    @staticmethod
    def distribution_check(
        data: pl.DataFrame,
        column: str,
        distribution_fn: Callable[[np.ndarray], float],
        threshold: float
    ) -> Dict[str, Any]:
        """
        Check if data distribution meets a specific criteria
        
        Args:
            data: DataFrame containing feature values
            column: Column name to check
            distribution_fn: Function that computes a metric on the distribution
            threshold: Threshold for the distribution metric
            
        Returns:
            Validation results
        """
        if column not in data.columns:
            return {
                "valid": False,
                "error": f"Column '{column}' not found in data"
            }
            
        # Convert to numpy array
        values = data.select(pl.col(column)).drop_nulls().to_numpy().flatten()
        
        if len(values) == 0:
            return {
                "valid": False,
                "error": f"No non-null values found in column '{column}'"
            }
            
        # Compute distribution metric
        metric = distribution_fn(values)
        
        results = {
            "column": column,
            "distribution_metric": metric,
            "threshold": threshold
        }
        
        # Validate distribution metric
        if metric > threshold:
            results["valid"] = True
        else:
            results["valid"] = False
            results["error"] = f"Distribution metric {metric:.4f} does not meet threshold {threshold}"
            
        return results
```

## Conclusion

This feature engineering framework provides a robust foundation for the NCAA Basketball Analytics project. It enables:

1. **Reproducibility**: Feature versions and clear lineage
2. **Efficiency**: Incremental calculations and optimal storage
3. **Flexibility**: Easy addition of new features and composite features
4. **Traceability**: Complete logging of feature computation
5. **Validation**: Built-in quality checks and testing
6. **Integration**: Seamless connection with Airflow, DuckDB, and the broader ML pipeline

The framework is designed to grow with the project, allowing for increasingly sophisticated features as the predictive models evolve.
