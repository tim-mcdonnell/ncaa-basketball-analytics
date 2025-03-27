"""
Tests for the feature dependency resolution system.

This module tests the functionality of the dependency resolution system, which is
responsible for identifying and resolving feature dependencies.
"""

import pytest

from src.features.dependencies import DependencyResolver, DependencyError
from src.features.registry import Feature, FeatureMetadata


class TestDependencyResolver:
    """Tests for the dependency resolution functionality."""

    def test_empty_dependencies(self):
        """Test resolving when no dependencies exist."""
        # Arrange
        resolver = DependencyResolver()
        feature = Feature(
            name="simple_feature",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature with no dependencies",
                category="test",
                tags=["test"],
            ),
            dependencies=[],
        )

        # Act
        resolved = resolver.resolve([feature])

        # Assert
        assert len(resolved) == 1
        assert resolved[0] == feature

    def test_simple_dependency_chain(self):
        """Test resolving a simple dependency chain."""
        # Arrange
        resolver = DependencyResolver()

        # Feature C depends on B depends on A
        feature_a = Feature(
            name="feature_a",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Base feature A",
                category="test",
                tags=["test"],
            ),
            dependencies=[],
        )

        feature_b = Feature(
            name="feature_b",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature B depends on A",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_a"],
        )

        feature_c = Feature(
            name="feature_c",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature C depends on B",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_b"],
        )

        # Act
        resolved = resolver.resolve([feature_c, feature_b, feature_a])

        # Assert
        # The order should be A, B, C (topological sort)
        assert len(resolved) == 3
        assert resolved[0] == feature_a
        assert resolved[1] == feature_b
        assert resolved[2] == feature_c

    def test_diamond_dependency(self):
        """Test resolving a diamond dependency pattern."""
        # Arrange
        resolver = DependencyResolver()

        # A <- B
        # ↑    ↑
        # └─ C ┘
        feature_a = Feature(
            name="feature_a",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Base feature A",
                category="test",
                tags=["test"],
            ),
            dependencies=[],
        )

        feature_b = Feature(
            name="feature_b",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature B depends on A",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_a"],
        )

        feature_c = Feature(
            name="feature_c",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature C depends on A",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_a"],
        )

        feature_d = Feature(
            name="feature_d",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature D depends on B and C",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_b", "feature_c"],
        )

        # Act - shuffle the order to test sorting
        resolved = resolver.resolve([feature_d, feature_a, feature_c, feature_b])

        # Assert
        assert len(resolved) == 4
        # A must come before B and C
        assert resolved.index(feature_a) < resolved.index(feature_b)
        assert resolved.index(feature_a) < resolved.index(feature_c)
        # B and C must come before D
        assert resolved.index(feature_b) < resolved.index(feature_d)
        assert resolved.index(feature_c) < resolved.index(feature_d)

    def test_detect_circular_dependency(self):
        """Test that circular dependencies are detected and raise an error."""
        # Arrange
        resolver = DependencyResolver()

        # Circular: A → B → C → A
        feature_a = Feature(
            name="feature_a",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature A depends on C",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_c"],
        )

        feature_b = Feature(
            name="feature_b",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature B depends on A",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_a"],
        )

        feature_c = Feature(
            name="feature_c",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature C depends on B",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_b"],
        )

        # Act & Assert
        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve([feature_a, feature_b, feature_c])

        assert "Circular dependency" in str(exc_info.value)

    def test_missing_dependency(self):
        """Test that missing dependencies are detected and raise an error."""
        # Arrange
        resolver = DependencyResolver()

        feature = Feature(
            name="feature_a",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature with missing dependency",
                category="test",
                tags=["test"],
            ),
            dependencies=["nonexistent_feature"],
        )

        # Act & Assert
        with pytest.raises(DependencyError) as exc_info:
            resolver.resolve([feature])

        assert "Missing dependency" in str(exc_info.value)
        assert "nonexistent_feature" in str(exc_info.value)

    def test_version_specific_dependencies(self):
        """Test resolving dependencies with specific versions."""
        # Arrange
        resolver = DependencyResolver()

        feature_a_v1 = Feature(
            name="feature_a",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature A v1",
                category="test",
                tags=["test"],
            ),
            dependencies=[],
        )

        feature_a_v2 = Feature(
            name="feature_a",
            version="2.0.0",
            metadata=FeatureMetadata(
                description="Feature A v2",
                category="test",
                tags=["test"],
            ),
            dependencies=[],
        )

        feature_b = Feature(
            name="feature_b",
            version="1.0.0",
            metadata=FeatureMetadata(
                description="Feature B depends on A v2",
                category="test",
                tags=["test"],
            ),
            dependencies=["feature_a@2.0.0"],  # Version-specific dependency
        )

        # Act
        resolved = resolver.resolve([feature_b, feature_a_v1, feature_a_v2])

        # Assert
        assert len(resolved) == 2  # Should only include A v2 and B
        assert feature_a_v2 in resolved
        assert feature_b in resolved
        assert feature_a_v1 not in resolved
        assert resolved.index(feature_a_v2) < resolved.index(feature_b)
