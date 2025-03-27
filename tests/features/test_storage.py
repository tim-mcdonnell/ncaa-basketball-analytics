"""
Tests for the feature storage system.

This module tests the functionality of the feature storage system, which is responsible
for persisting and retrieving feature values in DuckDB.
"""

import pytest
import polars as pl
import duckdb
import tempfile
import os

from src.features.storage.writer import FeatureWriter
from src.features.storage.reader import FeatureReader
from src.features.registry import Feature, FeatureMetadata


class TestFeatureStorage:
    """Tests for the feature storage functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        # Just create a temporary path but don't create the file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            db_path = temp_file.name
            # Close and remove the file - DuckDB will create it
            temp_file.close()
            os.unlink(db_path)

        # Add .duckdb extension
        db_path = f"{db_path}.duckdb"

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def sample_feature(self):
        """Create a sample feature for testing."""
        return Feature(
            name="test_feature",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Test feature for storage",
                category="test",
                tags=["test", "storage"],
            ),
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample feature data for testing."""
        return pl.DataFrame({"entity_id": ["team1", "team2", "team3"], "value": [10.5, 20.7, 15.2]})

    def test_feature_writer_initialization(self, temp_db_path):
        """Test initializing the feature writer."""
        # Arrange & Act
        writer = FeatureWriter(db_path=temp_db_path)

        # Assert
        assert writer.db_path == temp_db_path
        assert os.path.exists(temp_db_path)

    def test_feature_writer_store_feature(self, temp_db_path, sample_feature, sample_data):
        """Test storing feature values in the database."""
        # Arrange
        writer = FeatureWriter(db_path=temp_db_path)

        # Act
        writer.store(
            feature=sample_feature,
            data=sample_data,
            entity_type="team",
            entity_id_column="entity_id",
            value_column="value",
        )

        # Assert - Check if data was stored by querying
        conn = duckdb.connect(temp_db_path)
        result = conn.execute("""
            SELECT * FROM feature_values
            WHERE feature_name = 'test_feature'
            AND feature_version = '1.0.0'
            ORDER BY entity_id
        """).fetchall()
        conn.close()

        assert len(result) == 3
        assert result[0][2] == "team1"  # entity_id
        assert result[0][3] == "team"  # entity_type
        assert float(result[0][5]) == 10.5  # value

    def test_feature_reader_initialization(self, temp_db_path):
        """Test initializing the feature reader."""
        # Arrange & Act
        reader = FeatureReader(db_path=temp_db_path)

        # Assert
        assert reader.db_path == temp_db_path

    def test_feature_reader_get_values(self, temp_db_path, sample_feature, sample_data):
        """Test retrieving feature values from the database."""
        # Arrange - Store data first
        writer = FeatureWriter(db_path=temp_db_path)
        writer.store(
            feature=sample_feature,
            data=sample_data,
            entity_type="team",
            entity_id_column="entity_id",
            value_column="value",
        )

        reader = FeatureReader(db_path=temp_db_path)

        # Act - Retrieve the data
        result = reader.get_values(
            feature_name=sample_feature.name,
            feature_version=sample_feature.version,
            entity_type="team",
        )

        # Assert
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 3
        assert "entity_id" in result.columns
        assert "value" in result.columns
        assert result.filter(pl.col("entity_id") == "team1").select("value").item() == 10.5

    def test_feature_reader_get_values_for_entity(self, temp_db_path, sample_feature, sample_data):
        """Test retrieving feature values for a specific entity."""
        # Arrange - Store data first
        writer = FeatureWriter(db_path=temp_db_path)
        writer.store(
            feature=sample_feature,
            data=sample_data,
            entity_type="team",
            entity_id_column="entity_id",
            value_column="value",
        )

        reader = FeatureReader(db_path=temp_db_path)

        # Act - Retrieve data for a specific entity
        value = reader.get_value(
            feature_name=sample_feature.name,
            feature_version=sample_feature.version,
            entity_type="team",
            entity_id="team2",
        )

        # Assert
        assert value == 20.7

    def test_feature_writer_update_feature(self, temp_db_path, sample_feature):
        """Test updating existing feature values."""
        # Arrange
        writer = FeatureWriter(db_path=temp_db_path)

        # Initial data
        initial_data = pl.DataFrame({"entity_id": ["team1", "team2"], "value": [10.5, 20.7]})

        # Store initial data
        writer.store(
            feature=sample_feature,
            data=initial_data,
            entity_type="team",
            entity_id_column="entity_id",
            value_column="value",
        )

        # Updated data
        updated_data = pl.DataFrame(
            {"entity_id": ["team1", "team2", "team3"], "value": [15.0, 25.0, 30.0]}
        )

        # Act - Update with new data
        writer.store(
            feature=sample_feature,
            data=updated_data,
            entity_type="team",
            entity_id_column="entity_id",
            value_column="value",
            update_existing=True,
        )

        # Assert
        reader = FeatureReader(db_path=temp_db_path)
        result = reader.get_values(
            feature_name=sample_feature.name,
            feature_version=sample_feature.version,
            entity_type="team",
        )

        # Should have 3 entries with updated values
        assert result.shape[0] == 3
        assert result.filter(pl.col("entity_id") == "team1").select("value").item() == 15.0
        assert result.filter(pl.col("entity_id") == "team2").select("value").item() == 25.0
        assert result.filter(pl.col("entity_id") == "team3").select("value").item() == 30.0

    def test_feature_storage_with_timestamps(self, temp_db_path, sample_feature):
        """Test storing and retrieving features with timestamps."""
        # Arrange
        writer = FeatureWriter(db_path=temp_db_path)

        # Data with timestamps
        data = pl.DataFrame(
            {
                "entity_id": ["team1", "team1", "team1"],
                "value": [10.0, 15.0, 20.0],
                "timestamp": ["2023-01-01", "2023-02-01", "2023-03-01"],
            }
        )

        # Act - Store with timestamps
        writer.store(
            feature=sample_feature,
            data=data,
            entity_type="team",
            entity_id_column="entity_id",
            value_column="value",
            timestamp_column="timestamp",
        )

        # Assert
        reader = FeatureReader(db_path=temp_db_path)

        # Get latest value
        latest = reader.get_value(
            feature_name=sample_feature.name,
            feature_version=sample_feature.version,
            entity_type="team",
            entity_id="team1",
        )
        assert latest == 20.0  # Should get the most recent value

        # Get value at a specific time
        value_feb = reader.get_value(
            feature_name=sample_feature.name,
            feature_version=sample_feature.version,
            entity_type="team",
            entity_id="team1",
            as_of="2023-02-15",  # Should get the February value
        )
        assert value_feb == 15.0

        # Get time series
        time_series = reader.get_time_series(
            feature_name=sample_feature.name,
            feature_version=sample_feature.version,
            entity_type="team",
            entity_id="team1",
        )

        assert time_series.shape[0] == 3
        assert "timestamp" in time_series.columns
        assert "value" in time_series.columns

    def test_feature_storage_metadata(self, temp_db_path, sample_feature, sample_data):
        """Test storing and retrieving feature metadata."""
        # Arrange
        writer = FeatureWriter(db_path=temp_db_path)

        # Act - Store feature with metadata
        writer.store(
            feature=sample_feature,
            data=sample_data,
            entity_type="team",
            entity_id_column="entity_id",
            value_column="value",
        )

        # Also store feature metadata
        writer.store_metadata(sample_feature)

        # Assert
        reader = FeatureReader(db_path=temp_db_path)
        metadata = reader.get_feature_metadata(
            feature_name=sample_feature.name, feature_version=sample_feature.version
        )

        assert metadata is not None
        assert metadata["description"] == "Test feature for storage"
        assert metadata["category"] == "test"
        assert "test" in metadata["tags"]
        assert "storage" in metadata["tags"]
