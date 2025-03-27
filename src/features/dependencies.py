"""
Dependency resolution for features.

This module contains the DependencyResolver class, which is responsible for
resolving dependencies between features and ensuring they are computed in the
correct order.
"""

from typing import List, Dict, Set, Optional, Tuple
import re

from src.features.registry import Feature


class DependencyError(Exception):
    """Exception raised for dependency resolution errors."""

    pass


class DependencyResolver:
    """
    Resolver for feature dependencies.

    This class provides methods to resolve dependencies between features and
    determine the correct order for feature computation.
    """

    def __init__(self):
        """Initialize a dependency resolver."""
        pass

    def resolve(self, features: List[Feature]) -> List[Feature]:
        """
        Resolve dependencies between features and determine computation order.

        Args:
            features: List of features to resolve dependencies for

        Returns:
            List of features in the order they should be computed

        Raises:
            DependencyError: If there is a circular dependency or a dependency cannot be found
        """
        # First, we need to determine which versions of features to include
        # when multiple versions of the same feature are available
        included_features = self._determine_included_features(features)

        # Build a map of feature names and keys to features for quick lookup
        feature_map: Dict[str, Feature] = {}
        for feature in included_features:
            feature_map[feature.key] = feature

            # Also map by name (without version) for non-versioned dependencies
            if feature.name not in feature_map:
                feature_map[feature.name] = feature

        # Build dependency graph
        graph: Dict[str, List[str]] = {}
        for feature in included_features:
            graph[feature.key] = []
            for dep in feature.dependencies:
                # Check if the dependency has a version specified
                if "@" in dep:
                    # Use exact version
                    dep_key = dep
                else:
                    # Use dependency name only
                    dep_key = dep

                # Make sure the dependency exists
                if dep_key not in feature_map:
                    raise DependencyError(
                        f"Missing dependency: {dep_key} required by {feature.key}"
                    )

                # Add to graph
                graph[feature.key].append(feature_map[dep_key].key)

        # Detect cycles
        visited: Set[str] = set()
        temp_visited: Set[str] = set()

        def dfs_cycle_detection(node: str) -> None:
            """
            Depth-first search to detect cycles in the dependency graph.

            Args:
                node: Current node in the graph

            Raises:
                DependencyError: If a cycle is detected
            """
            if node in temp_visited:
                # Cycle detected
                raise DependencyError(f"Circular dependency detected involving {node}")

            if node in visited:
                # Already processed
                return

            temp_visited.add(node)

            for neighbor in graph.get(node, []):
                dfs_cycle_detection(neighbor)

            temp_visited.remove(node)
            visited.add(node)

        # Topological sort implementation
        visited = set()
        result = []

        def topo_sort(node: str) -> None:
            """
            Perform topological sort starting from the given node.

            Args:
                node: Starting node for sort
            """
            if node in visited:
                return

            visited.add(node)

            # Visit all dependencies first
            for dependency in graph.get(node, []):
                topo_sort(dependency)

            # Add current node after all dependencies
            result.append(feature_map[node])

        # Check for cycles first
        for node in graph:
            if node not in visited:
                dfs_cycle_detection(node)

        # Reset visited set for topological sort
        visited = set()

        # Perform topological sort
        for node in graph:
            if node not in visited:
                topo_sort(node)

        return result

    def _determine_included_features(self, features: List[Feature]) -> List[Feature]:
        """
        Determine which features to include when multiple versions of the same feature exist.

        Args:
            features: List of features to analyze

        Returns:
            List of features to include in the dependency resolution
        """
        # Collect feature names that have specific version requirements
        version_requirements: Dict[str, Set[str]] = {}

        # First pass: collect all version-specific dependencies
        for feature in features:
            for dep in feature.dependencies:
                if "@" in dep:
                    name, version = self.parse_dependency(dep)
                    if name not in version_requirements:
                        version_requirements[name] = set()
                    if version:
                        version_requirements[name].add(version)

        # Collect features by name for easy lookup
        features_by_name: Dict[str, Dict[str, Feature]] = {}
        for feature in features:
            if feature.name not in features_by_name:
                features_by_name[feature.name] = {}
            features_by_name[feature.name][feature.version] = feature

        # Determine which features to include
        included_features = []
        included_feature_names = set()

        # First, add all features with specific version requirements
        for name, versions in version_requirements.items():
            if name in features_by_name:
                for version in versions:
                    if version in features_by_name[name]:
                        included_features.append(features_by_name[name][version])
                        included_feature_names.add(f"{name}@{version}")

        # Then, add all features that don't have specific version requirements
        # or are not dependencies with a specific version
        for feature in features:
            feature_key = f"{feature.name}@{feature.version}"
            # If this is not a feature with a specific version requirement
            if (
                feature.name not in version_requirements
                or feature_key not in included_feature_names
            ):
                # Check if we already included a different version of this feature
                # due to a specific version requirement
                if (
                    feature.name in version_requirements
                    and len(version_requirements[feature.name]) > 0
                ):
                    # Skip this feature if we already have a specific version
                    continue

                # Add this feature
                included_features.append(feature)

        return included_features

    def parse_dependency(self, dep_str: str) -> Tuple[str, Optional[str]]:
        """
        Parse a dependency string into name and version components.

        Args:
            dep_str: Dependency string in the format "name" or "name@version"

        Returns:
            Tuple of (name, version) where version may be None
        """
        match = re.match(r"([^@]+)(?:@(.+))?", dep_str)
        if not match:
            return dep_str, None

        name, version = match.groups()
        return name, version
