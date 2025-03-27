"""
Tests for the base feature classes.

This module tests the functionality of the base feature classes that provide
the foundation for feature calculation and caching.
"""

from typing import Dict, Any

from src.features.base import BaseFeature, CachedFeature
from src.features.registry import FeatureMetadata


class SimpleFeature(BaseFeature):
    """A simple feature implementation for testing."""

    def __init__(self, name: str = "simple_feature", version: str = "1.0.0"):
        """Initialize with default name and version."""
        metadata = FeatureMetadata(
            description="Simple test feature", category="test", tags=["test"]
        )
        super().__init__(name=name, version=version, metadata=metadata)

    def compute(self, data: Dict[str, Any]) -> Any:
        """Simple compute method that returns double the input value."""
        return data.get("value", 0) * 2


class DependentFeature(BaseFeature):
    """A feature with dependencies for testing."""

    def __init__(self, name: str = "dependent_feature", version: str = "1.0.0"):
        """Initialize with default name, version, and dependencies."""
        metadata = FeatureMetadata(
            description="Feature with dependencies", category="test", tags=["test"]
        )
        dependencies = ["simple_feature"]
        super().__init__(name=name, version=version, metadata=metadata, dependencies=dependencies)

    def compute(self, data: Dict[str, Any]) -> Any:
        """Compute method that depends on simple_feature."""
        simple_value = data.get("simple_feature", 0)
        return simple_value + 10


class TestBaseFeature:
    """Tests for the BaseFeature class."""

    def test_feature_initialization(self):
        """Test the initialization of a feature."""
        # Arrange & Act
        feature = SimpleFeature()

        # Assert
        assert feature.name == "simple_feature"
        assert feature.version == "1.0.0"
        assert feature.metadata.description == "Simple test feature"
        assert feature.metadata.category == "test"
        assert "test" in feature.metadata.tags
        assert feature.dependencies == []

    def test_feature_compute(self):
        """Test the compute method of a feature."""
        # Arrange
        feature = SimpleFeature()
        data = {"value": 5}

        # Act
        result = feature.compute(data)

        # Assert
        assert result == 10

    def test_feature_with_dependencies(self):
        """Test a feature with dependencies."""
        # Arrange
        dependent_feature = DependentFeature()

        # Assert
        assert dependent_feature.dependencies == ["simple_feature"]

    def test_feature_key(self):
        """Test the feature key property."""
        # Arrange
        feature = SimpleFeature(name="test_feature", version="2.1.0")

        # Act
        key = feature.key

        # Assert
        assert key == "test_feature@2.1.0"

    def test_feature_equality(self):
        """Test that features with the same name and version are equal."""
        # Arrange
        feature1 = SimpleFeature(name="test_feature", version="1.0.0")
        feature2 = SimpleFeature(name="test_feature", version="1.0.0")
        feature3 = SimpleFeature(name="different_feature", version="1.0.0")
        feature4 = SimpleFeature(name="test_feature", version="2.0.0")

        # Assert
        assert feature1 == feature2
        assert feature1 != feature3
        assert feature1 != feature4

    def test_feature_hash(self):
        """Test that features with the same name and version have the same hash."""
        # Arrange
        feature1 = SimpleFeature(name="test_feature", version="1.0.0")
        feature2 = SimpleFeature(name="test_feature", version="1.0.0")

        # Act
        hash1 = hash(feature1)
        hash2 = hash(feature2)

        # Assert
        assert hash1 == hash2

        # Also test in a set
        feature_set = {feature1, feature2}
        assert len(feature_set) == 1


class TestCachedFeature:
    """Tests for the CachedFeature class."""

    class SimpleCachedFeature(CachedFeature):
        """A simple cached feature implementation for testing."""

        def __init__(self, name: str = "cached_feature", version: str = "1.0.0"):
            """Initialize with default name and version."""
            metadata = FeatureMetadata(
                description="Simple cached test feature", category="test", tags=["test"]
            )
            super().__init__(name=name, version=version, metadata=metadata)
            self.compute_count = 0

        def _compute_uncached(self, data: Dict[str, Any]) -> Any:
            """Compute method that increments a counter for testing cache hits."""
            self.compute_count += 1
            return data.get("value", 0) * 3

    def test_cached_feature_initialization(self):
        """Test the initialization of a cached feature."""
        # Arrange & Act
        feature = self.SimpleCachedFeature()

        # Assert
        assert feature.name == "cached_feature"
        assert feature.version == "1.0.0"
        assert feature.metadata.description == "Simple cached test feature"
        assert feature.cache_enabled is True

    def test_cached_feature_compute(self):
        """Test that computed values are cached."""
        # Arrange
        feature = self.SimpleCachedFeature()
        data = {"value": 5}

        # Act - compute twice with the same data
        result1 = feature.compute(data)
        result2 = feature.compute(data)

        # Assert
        assert result1 == 15
        assert result2 == 15
        assert feature.compute_count == 1  # Should only compute once

    def test_cached_feature_invalidate_cache(self):
        """Test that cache invalidation forces recomputation."""
        # Arrange
        feature = self.SimpleCachedFeature()
        data = {"value": 5}

        # Act - compute, invalidate cache, compute again
        result1 = feature.compute(data)
        feature.invalidate_cache()
        result2 = feature.compute(data)

        # Assert
        assert result1 == 15
        assert result2 == 15
        assert feature.compute_count == 2  # Should compute twice

    def test_cached_feature_disable_cache(self):
        """Test that disabling the cache forces recomputation."""
        # Arrange
        feature = self.SimpleCachedFeature()
        data = {"value": 5}

        # Act - compute with cache, disable cache, compute again
        result1 = feature.compute(data)
        feature.cache_enabled = False
        result2 = feature.compute(data)

        # Assert
        assert result1 == 15
        assert result2 == 15
        assert feature.compute_count == 2  # Should compute twice

    def test_cached_feature_batch_compute(self):
        """Test batch computation with caching."""
        # Arrange
        feature = self.SimpleCachedFeature()
        batch_data = [{"id": 1, "value": 5}, {"id": 2, "value": 10}, {"id": 3, "value": 15}]

        # Act
        results = feature.batch_compute(batch_data, id_field="id")

        # Assert
        assert len(results) == 3
        assert results[1] == 15  # 5 * 3
        assert results[2] == 30  # 10 * 3
        assert results[3] == 45  # 15 * 3
        assert feature.compute_count == 3  # Should compute once per item
