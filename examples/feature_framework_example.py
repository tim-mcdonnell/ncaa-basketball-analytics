"""
NCAA Basketball Analytics - Feature Framework Example

This script demonstrates the basic usage of the feature engineering framework.
It shows how to create features, compute them, and store/retrieve them from the database.

NOTE: This is a standalone example that doesn't require importing from the project.
"""

import os
import polars as pl
import json
import duckdb
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


# Use the same database as data_storage_example.py
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "example.duckdb")

# Check if database exists, create it if not
if not os.path.exists(db_path):
    print(f"Database does not exist at {db_path}. Creating a new one.")
    conn = duckdb.connect(db_path)
    conn.close()

print(f"Using database at: {db_path}")


# Step 1: Define the feature framework classes


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""

    description: str
    category: str
    tags: List[str]


class Feature:
    """Feature definition class."""

    def __init__(
        self,
        name: str,
        version: str,
        metadata: FeatureMetadata,
        dependencies: Optional[List[str]] = None,
    ):
        self.name = name
        self.version = version
        self.metadata = metadata
        self.dependencies = dependencies or []

    @property
    def key(self) -> str:
        """Get the feature key (name@version)."""
        return f"{self.name}@{self.version}"


class BaseFeature:
    """Base class for all features."""

    def __init__(
        self,
        name: str,
        version: str,
        metadata: FeatureMetadata,
        dependencies: Optional[List[str]] = None,
    ):
        self.name = name
        self.version = version
        self.metadata = metadata
        self.dependencies = dependencies or []

    @property
    def key(self) -> str:
        """Get the feature key (name@version)."""
        return f"{self.name}@{self.version}"

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compute the feature value."""
        raise NotImplementedError("Subclasses must implement compute method")


class FeatureRegistry:
    """Registry for features."""

    def __init__(self):
        self._features = {}

    def register(self, feature: BaseFeature) -> None:
        """Register a feature."""
        self._features[feature.key] = feature

    def get(self, name: str, version: Optional[str] = None) -> Optional[BaseFeature]:
        """Get a feature by name and version."""
        if version:
            key = f"{name}@{version}"
            return self._features.get(key)

        # Return the latest version if no version specified
        candidates = [f for k, f in self._features.items() if k.startswith(f"{name}@")]
        if not candidates:
            return None

        # Sort by version and return latest
        return sorted(candidates, key=lambda f: f.version)[-1]

    def list_features(self) -> List[BaseFeature]:
        """List all registered features."""
        return list(self._features.values())


class DependencyResolver:
    """Resolver for feature dependencies."""

    def resolve(self, features: List[BaseFeature]) -> List[BaseFeature]:
        """Resolve dependencies and return computation order."""
        # Build dependency graph
        graph = {}
        feature_map = {}

        for feature in features:
            graph[feature.key] = []
            feature_map[feature.key] = feature

            # Also map by name for non-versioned dependencies
            feature_map[feature.name] = feature

        # Add dependencies to graph
        for feature in features:
            for dep in feature.dependencies:
                if "@" in dep:
                    # Exact version specified
                    dep_key = dep
                else:
                    # Use latest version
                    dep_key = feature_map[dep].key

                graph[feature.key].append(dep_key)

        # Topological sort
        visited = set()
        result = []

        def visit(node):
            if node in visited:
                return

            visited.add(node)

            for dep in graph.get(node, []):
                visit(dep)

            result.append(feature_map[node])

        # Visit all nodes
        for node in graph:
            if node not in visited:
                visit(node)

        return result


class FeatureWriter:
    """Writer for storing feature values."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        conn = duckdb.connect(self.db_path)

        # Create sequence for auto-incrementing IDs
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS feature_values_id_seq;
        """)

        # Create feature values table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                id INTEGER PRIMARY KEY DEFAULT nextval('feature_values_id_seq'),
                feature_name VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                entity_type VARCHAR NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                value VARCHAR NOT NULL,
                feature_version VARCHAR NOT NULL
            )
        """)

        # Create feature metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name VARCHAR NOT NULL,
                feature_version VARCHAR NOT NULL,
                metadata JSON NOT NULL,
                PRIMARY KEY (feature_name, feature_version)
            )
        """)

        # Create indexes for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_values_lookup
            ON feature_values (feature_name, feature_version, entity_type, entity_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_values_timestamp
            ON feature_values (feature_name, feature_version, entity_id, timestamp)
        """)

        conn.close()

    def store(
        self,
        feature: Feature,
        data: pl.DataFrame,
        entity_type: str,
        entity_id_column: str,
        value_column: str,
        timestamp_column: Optional[str] = None,
        update_existing: bool = False,
    ) -> None:
        """Store feature values in the database."""
        # Make sure required columns exist
        if entity_id_column not in data.columns:
            raise ValueError(f"Entity ID column '{entity_id_column}' not in DataFrame")

        if value_column not in data.columns:
            raise ValueError(f"Value column '{value_column}' not in DataFrame")

        if timestamp_column and timestamp_column not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not in DataFrame")

        # Convert to records for insertion
        records = []
        for row in data.rows(named=True):
            entity_id = row[entity_id_column]
            value = row[value_column]

            record = {
                "feature_name": feature.name,
                "feature_version": feature.version,
                "entity_id": str(entity_id),
                "entity_type": entity_type,
                "value": json.dumps(value) if not isinstance(value, str) else value,
            }

            # Add timestamp if provided
            if timestamp_column:
                record["timestamp"] = row[timestamp_column]

            records.append(record)

        # Insert or update records
        conn = duckdb.connect(self.db_path)

        if update_existing:
            # First delete existing values for this feature/entity combination
            conn.execute(
                """
                DELETE FROM feature_values
                WHERE feature_name = ? AND feature_version = ? AND entity_type = ?
                AND entity_id IN (SELECT DISTINCT entity_id FROM feature_values
                                 WHERE feature_name = ? AND feature_version = ? AND entity_type = ?)
            """,
                (
                    feature.name,
                    feature.version,
                    entity_type,
                    feature.name,
                    feature.version,
                    entity_type,
                ),
            )

        # Insert new values
        for record in records:
            if timestamp_column:
                conn.execute(
                    """
                    INSERT INTO feature_values (feature_name, feature_version, entity_id, entity_type, timestamp, value)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        record["feature_name"],
                        record["feature_version"],
                        record["entity_id"],
                        record["entity_type"],
                        record["timestamp"],
                        record["value"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO feature_values (feature_name, feature_version, entity_id, entity_type, value)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        record["feature_name"],
                        record["feature_version"],
                        record["entity_id"],
                        record["entity_type"],
                        record["value"],
                    ),
                )

        conn.close()

    def store_metadata(self, feature: Feature) -> None:
        """Store feature metadata in the database."""
        # Convert metadata to JSON
        metadata_dict = {
            "description": feature.metadata.description,
            "category": feature.metadata.category,
            "tags": feature.metadata.tags,
            "dependencies": feature.dependencies,
        }

        metadata_json = json.dumps(metadata_dict)

        # Store in database
        conn = duckdb.connect(self.db_path)

        # Use upsert to handle existing metadata
        conn.execute(
            """
            DELETE FROM feature_metadata
            WHERE feature_name = ? AND feature_version = ?
        """,
            (feature.name, feature.version),
        )

        conn.execute(
            """
            INSERT INTO feature_metadata (feature_name, feature_version, metadata)
            VALUES (?, ?, ?)
        """,
            (feature.name, feature.version, metadata_json),
        )

        conn.close()


class FeatureReader:
    """Reader for retrieving feature values."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _convert_value(self, value_str: str) -> Any:
        """Convert a string value to the appropriate type."""
        if not isinstance(value_str, str):
            return value_str

        # Try to convert JSON
        if value_str.startswith("[") or value_str.startswith("{"):
            return json.loads(value_str)

        # Try to convert to numeric
        try:
            # Check if it's an integer
            if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
                return int(value_str)
            # Check if it's a float (contains a decimal point)
            elif "." in value_str:
                return float(value_str)
        except (ValueError, TypeError):
            pass

        # Return as is if no conversion applies
        return value_str

    def get_values(
        self, feature_name: str, feature_version: str, entity_type: str, as_of: Optional[str] = None
    ) -> pl.DataFrame:
        """Get feature values for a specific feature."""
        conn = duckdb.connect(self.db_path)

        if as_of:
            # Get values as of a specific time
            query = """
                WITH latest_values AS (
                    SELECT
                        entity_id,
                        value,
                        MAX(timestamp) as max_timestamp
                    FROM feature_values
                    WHERE feature_name = ?
                    AND feature_version = ?
                    AND entity_type = ?
                    AND timestamp <= ?
                    GROUP BY entity_id, value
                )
                SELECT entity_id, value
                FROM latest_values
            """
            result = conn.execute(query, (feature_name, feature_version, entity_type, as_of)).pl()
        else:
            # Get latest values
            query = """
                WITH latest_timestamps AS (
                    SELECT
                        entity_id,
                        MAX(timestamp) as max_timestamp
                    FROM feature_values
                    WHERE feature_name = ?
                    AND feature_version = ?
                    AND entity_type = ?
                    GROUP BY entity_id
                )
                SELECT fv.entity_id, fv.value
                FROM feature_values fv
                JOIN latest_timestamps lt
                ON fv.entity_id = lt.entity_id AND fv.timestamp = lt.max_timestamp
                WHERE fv.feature_name = ?
                AND fv.feature_version = ?
                AND fv.entity_type = ?
            """
            result = conn.execute(
                query,
                (
                    feature_name,
                    feature_version,
                    entity_type,
                    feature_name,
                    feature_version,
                    entity_type,
                ),
            ).pl()

        conn.close()

        # Convert values from JSON string or numeric string if needed
        if not result.is_empty():
            result = result.with_columns(
                pl.col("value").map_elements(self._convert_value, return_dtype=pl.Object)
            )

        return result

    def get_value(
        self,
        feature_name: str,
        feature_version: str,
        entity_type: str,
        entity_id: str,
        as_of: Optional[str] = None,
    ) -> Any:
        """Get a single feature value for a specific entity."""
        conn = duckdb.connect(self.db_path)

        if as_of:
            # Get value as of a specific time
            query = """
                SELECT value
                FROM feature_values
                WHERE feature_name = ?
                AND feature_version = ?
                AND entity_type = ?
                AND entity_id = ?
                AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = conn.execute(
                query, (feature_name, feature_version, entity_type, entity_id, as_of)
            ).fetchone()
        else:
            # Get latest value
            query = """
                SELECT value
                FROM feature_values
                WHERE feature_name = ?
                AND feature_version = ?
                AND entity_type = ?
                AND entity_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = conn.execute(
                query, (feature_name, feature_version, entity_type, entity_id)
            ).fetchone()

        conn.close()

        if result is None:
            return None

        value = result[0]

        # Convert to appropriate type
        return self._convert_value(value)

    def get_feature_metadata(
        self, feature_name: str, feature_version: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific feature."""
        conn = duckdb.connect(self.db_path)

        query = """
            SELECT metadata
            FROM feature_metadata
            WHERE feature_name = ?
            AND feature_version = ?
        """

        result = conn.execute(query, (feature_name, feature_version)).fetchone()
        conn.close()

        if result is None:
            return None

        return json.loads(result[0])


# Step 2: Define some basic features
class TeamPointsPerGame(BaseFeature):
    """Calculate the average points per game for a team."""

    def __init__(self):
        metadata = FeatureMetadata(
            description="Average points scored per game by a team",
            category="team_offense",
            tags=["scoring", "team_stats"],
        )
        super().__init__("team_points_per_game", "1.0.0", metadata)

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Compute points per game from game data.

        Expected input data columns:
        - team_id: Unique identifier for the team
        - points: Points scored in each game
        """
        # Group by team_id and calculate mean points
        return data.group_by("team_id").agg(pl.col("points").mean().alias("value"))


class TeamDefensiveRating(BaseFeature):
    """Calculate the defensive rating for a team (points allowed per game)."""

    def __init__(self):
        metadata = FeatureMetadata(
            description="Average points allowed per game by a team",
            category="team_defense",
            tags=["defense", "team_stats"],
        )
        super().__init__("team_defensive_rating", "1.0.0", metadata)

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Compute defensive rating from game data.

        Expected input data columns:
        - team_id: Unique identifier for the team
        - opp_points: Points scored by opponents in each game
        """
        # Group by team_id and calculate mean opponent points
        return data.group_by("team_id").agg(pl.col("opp_points").mean().alias("value"))


class TeamNetRating(BaseFeature):
    """Calculate net rating (offensive rating minus defensive rating)."""

    def __init__(self):
        metadata = FeatureMetadata(
            description="Net rating (points per game minus points allowed per game)",
            category="team_overall",
            tags=["efficiency", "team_stats"],
        )
        # Note we're depending on the other two features
        super().__init__(
            "team_net_rating",
            "1.0.0",
            metadata,
            dependencies=["team_points_per_game@1.0.0", "team_defensive_rating@1.0.0"],
        )

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Compute net rating from offensive and defensive ratings.

        Expected input data columns:
        - team_id: Unique identifier for the team
        - team_points_per_game: Points per game (from the dependency)
        - team_defensive_rating: Points allowed per game (from the dependency)
        """
        return data.with_columns(
            (pl.col("team_points_per_game") - pl.col("team_defensive_rating")).alias("value")
        )


# Step 3: Create some sample data
def create_sample_data() -> pl.DataFrame:
    """Create sample game data for demonstration."""
    return pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5, 6],
            "date": [
                "2023-01-01",
                "2023-01-05",
                "2023-01-10",
                "2023-01-15",
                "2023-01-20",
                "2023-01-25",
            ],
            "team_id": ["team1", "team1", "team1", "team2", "team2", "team2"],
            "points": [75, 82, 68, 90, 65, 78],
            "opp_team_id": ["team2", "team3", "team4", "team1", "team3", "team4"],
            "opp_points": [70, 75, 60, 75, 70, 65],
        }
    )


# Step 4: Set up the feature registry
registry = FeatureRegistry()

# Register our features
ppg_feature = TeamPointsPerGame()
def_rating_feature = TeamDefensiveRating()
net_rating_feature = TeamNetRating()

registry.register(ppg_feature)
registry.register(def_rating_feature)
registry.register(net_rating_feature)

print(f"Registered features: {[f.name for f in registry.list_features()]}")


# Step 5: Resolve feature dependencies
resolver = DependencyResolver()
game_data = create_sample_data()

# For net rating, we need to compute the dependencies first
compute_order = resolver.resolve(features=[net_rating_feature, ppg_feature, def_rating_feature])

print(f"Computation order: {[f.name for f in compute_order]}")


# Step 6: Compute the features in order
writer = FeatureWriter(db_path=db_path)
feature_data = {}  # To store intermediate results

for feature in compute_order:
    print(f"Computing feature: {feature.name}")

    if feature.name == "team_points_per_game":
        # Compute points per game directly from game data
        result = feature.compute(game_data)
        feature_data[feature.name] = result
    elif feature.name == "team_defensive_rating":
        # Compute defensive rating directly from game data
        result = feature.compute(game_data)
        feature_data[feature.name] = result
    elif feature.name == "team_net_rating":
        # For net rating, we need to join the previously computed features
        # Get the data from our previous computations
        ppg_df = feature_data["team_points_per_game"].rename({"value": "team_points_per_game"})
        def_df = feature_data["team_defensive_rating"].rename({"value": "team_defensive_rating"})

        # Join the data
        joined_data = ppg_df.join(def_df, on="team_id")

        # Now compute the net rating
        result = feature.compute(joined_data)

    # Store in the database
    writer.store(
        feature=Feature(name=feature.name, version=feature.version, metadata=feature.metadata),
        data=result,
        entity_type="team",
        entity_id_column="team_id",
        value_column="value",
    )

    # Also store metadata
    writer.store_metadata(
        Feature(
            name=feature.name,
            version=feature.version,
            metadata=feature.metadata,
            dependencies=feature.dependencies,
        )
    )


# Step 7: Read back the features from storage
reader = FeatureReader(db_path=db_path)

# Get all teams' points per game
ppg_values = reader.get_values(
    feature_name="team_points_per_game", feature_version="1.0.0", entity_type="team"
)
print("\nTeam Points Per Game:")
print(ppg_values)

# Get net rating for a specific team
team1_net_rating = reader.get_value(
    feature_name="team_net_rating", feature_version="1.0.0", entity_type="team", entity_id="team1"
)
print(f"\nTeam1 Net Rating: {team1_net_rating}")

# Get feature metadata
net_rating_metadata = reader.get_feature_metadata(
    feature_name="team_net_rating", feature_version="1.0.0"
)
print("\nNet Rating Metadata:")
print(net_rating_metadata)

# Cleanup
print(f"\nExample complete. The feature database is at: {db_path}")
print("You can delete it manually when done exploring.")
