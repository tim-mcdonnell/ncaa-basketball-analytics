"""
Feature registry module.

This module defines the Feature class and the FeatureRegistry class, which is
responsible for managing feature definitions, metadata, and dependencies.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FeatureMetadata:
    """
    Metadata for a feature.

    Attributes:
        description: Descriptive text about the feature
        category: Category the feature belongs to
        tags: List of tags for the feature
    """

    description: str
    category: str
    tags: List[str]


class Feature:
    """
    Feature class representing a computable feature.

    Attributes:
        name: Unique name of the feature
        version: Version of the feature
        metadata: Additional information about the feature
        dependencies: List of feature names this feature depends on
    """

    def __init__(
        self,
        name: str,
        version: str,
        metadata: FeatureMetadata,
        dependencies: Optional[List[str]] = None,
    ):
        """
        Initialize a feature.

        Args:
            name: Unique name of the feature
            version: Version of the feature
            metadata: Additional information about the feature
            dependencies: List of feature names this feature depends on
        """
        self.name = name
        self.version = version
        self.metadata = metadata
        self.dependencies = dependencies or []

    @property
    def key(self) -> str:
        """
        Get the feature key, which is a combination of name and version.

        Returns:
            String in the format "name@version"
        """
        return f"{self.name}@{self.version}"

    def __eq__(self, other):
        """Check if two features are equal based on name and version."""
        if not isinstance(other, Feature):
            return False
        return self.name == other.name and self.version == other.version

    def __hash__(self):
        """Generate hash based on name and version."""
        return hash((self.name, self.version))


class FeatureRegistry:
    """
    Registry for managing features.

    This class provides methods to register, retrieve, and list features.
    """

    def __init__(self):
        """Initialize an empty feature registry."""
        self._features: Dict[str, Dict[str, Feature]] = {}
        self._latest_versions: Dict[str, str] = {}

    def register(self, feature: Feature) -> None:
        """
        Register a feature in the registry.

        Args:
            feature: The feature to register
        """
        if feature.name not in self._features:
            self._features[feature.name] = {}

        self._features[feature.name][feature.version] = feature

        # Update latest version if this is newer
        if feature.name not in self._latest_versions or self._is_newer_version(
            feature.version, self._latest_versions[feature.name]
        ):
            self._latest_versions[feature.name] = feature.version

    def get(self, name: str, version: Optional[str] = None) -> Optional[Feature]:
        """
        Get a feature by name and optionally version.

        Args:
            name: Name of the feature to retrieve
            version: Optional version to retrieve. If None, returns the latest version.

        Returns:
            The Feature object if found, None otherwise
        """
        if name not in self._features:
            return None

        if version is None:
            version = self._latest_versions.get(name)
            if version is None:
                return None

        return self._features[name].get(version)

    def list_features(
        self, category: Optional[str] = None, tag: Optional[str] = None
    ) -> List[Feature]:
        """
        List all registered features, optionally filtered by category or tag.

        Args:
            category: Optional category to filter by
            tag: Optional tag to filter by

        Returns:
            List of Feature objects
        """
        result = []

        for name, versions in self._features.items():
            # Use the latest version of each feature
            latest_version = self._latest_versions.get(name)
            if latest_version is None:
                continue

            feature = versions[latest_version]

            # Apply filters if specified
            if category is not None and feature.metadata.category != category:
                continue

            if tag is not None and tag not in feature.metadata.tags:
                continue

            result.append(feature)

        return result

    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """
        Compare two version strings to determine if version1 is newer than version2.

        Args:
            version1: First version string
            version2: Second version string

        Returns:
            True if version1 is newer than version2, False otherwise
        """
        # Split version into components (assumes semantic versioning)
        v1_parts = [int(x) for x in version1.split(".")]
        v2_parts = [int(x) for x in version2.split(".")]

        # Pad with zeros if needed
        while len(v1_parts) < len(v2_parts):
            v1_parts.append(0)
        while len(v2_parts) < len(v1_parts):
            v2_parts.append(0)

        # Compare component by component
        for i in range(len(v1_parts)):
            if v1_parts[i] > v2_parts[i]:
                return True
            if v1_parts[i] < v2_parts[i]:
                return False

        # If all components are equal, versions are the same
        return False
