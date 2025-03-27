# Adding New Features

This guide explains how to implement new features in the NCAA Basketball Analytics system.

## Feature Engineering Framework

The project uses a structured framework for feature engineering, following the architecture defined in the [Feature Engineering](../architecture/feature-engineering.md) document.

## Feature Types

The system supports several types of features:

- **Base Features**: Simple features derived directly from source data
- **Derived Features**: Features that build on other features
- **Composite Features**: Complex features that combine multiple inputs
- **Temporal Features**: Features that involve time-series calculations

## Creating a New Feature

### 1. Define Feature Requirements

First, clearly define:

- What the feature measures
- Why it's valuable for analysis or prediction
- What data it requires
- How it should be calculated

### 2. Create a Feature Class

Create a new class that extends the appropriate base class:

```python
# src/features/team_features.py
from src.features.base import BaseFeature
import polars as pl

class TeamScoringEfficiency(BaseFeature):
    """
    Measures a team's scoring efficiency as points per possession
    """

    def __init__(self, version=1):
        super().__init__(
            name="team_scoring_efficiency",
            version=version,
            description="Points scored per 100 possessions",
            dependencies=["team_points", "team_possessions"],
            category="team_offense",
            tags=["efficiency", "scoring"]
        )

    def _compute_impl(
        self,
        data: pl.DataFrame,
        db_conn=None,
        batch_id=None,
        incremental=True,
        **kwargs
    ) -> pl.DataFrame:
        # Implementation logic here
        result = (
            data.with_columns([
                (pl.col("points") * 100 / pl.col("possessions"))
                .alias(self.get_output_column_name())
            ])
        )
        return result

    def get_output_column_name(self) -> str:
        return f"team_scoring_efficiency_v{self.version}"

    def _validate_impl(self, data: pl.DataFrame) -> dict:
        feature_col = self.get_output_column_name()
        values = data[feature_col]

        return {
            "min_acceptable": 50,  # Minimum reasonable value
            "max_acceptable": 150,  # Maximum reasonable value
            "outliers": (
                (values < 50) | (values > 150)
            ).sum() / len(values) * 100
        }
```

### 3. Register the Feature

Add the feature to the feature registry:

```python
# src/features/registry.py
from src.features.team_features import TeamScoringEfficiency

def register_features():
    registry = FeatureRegistry()

    # Register existing features...

    # Register new feature
    registry.register(TeamScoringEfficiency())

    return registry
```

### 4. Create Tests

Create test cases for your feature:

```python
# tests/features/test_team_features.py
import polars as pl
import pytest
from src.features.team_features import TeamScoringEfficiency

def test_team_scoring_efficiency():
    # Create test data
    test_data = pl.DataFrame({
        "team_id": ["1", "2", "3"],
        "points": [80, 60, 90],
        "possessions": [70, 65, 75]
    })

    # Initialize feature
    feature = TeamScoringEfficiency()

    # Compute feature
    result = feature.compute(test_data)

    # Check output
    assert feature.get_output_column_name() in result.columns

    # Verify calculations
    expected = [80 * 100 / 70, 60 * 100 / 65, 90 * 100 / 75]
    actual = result[feature.get_output_column_name()].to_list()

    for e, a in zip(expected, actual):
        assert pytest.approx(e, 0.01) == a
```

### 5. Run the Feature in Airflow

Update the relevant Airflow DAG to include your feature:

```python
# airflow/dags/feature_engineering/team_features_dag.py
from src.features.team_features import TeamScoringEfficiency

# In your task definition:
def compute_team_features(**kwargs):
    # ...existing code...

    # Add your new feature
    features.append(TeamScoringEfficiency())

    # ...continue with feature computation...
```

## Feature Versioning

When updating an existing feature:

1. Increment the version number
2. Document the changes
3. Update tests to account for both versions

## Best Practices

- Follow the naming conventions established in existing features
- Document the feature thoroughly, especially its purpose and methodology
- Ensure efficient computation for large datasets
- Add appropriate validation logic
- Write comprehensive tests
