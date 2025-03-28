"""
Base feature classes.

This module contains the base classes for features, providing the foundation for
feature calculation and caching.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Hashable
import hashlib
import json
import datetime

from src.features.registry import FeatureMetadata


T = TypeVar("T")  # Type for feature value


class BaseFeature(ABC):
    """
    Base class for all features.

    This class defines the common interface for all features.

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

    @abstractmethod
    def compute(self, data: Dict[str, Any]) -> Any:
        """
        Compute the feature value.

        Args:
            data: Dictionary containing data needed for computation

        Returns:
            Computed feature value
        """
        pass

    def batch_compute(
        self, batch_data: List[Dict[str, Any]], id_field: str = "id"
    ) -> Dict[Hashable, Any]:
        """
        Compute feature values for a batch of data.

        Args:
            batch_data: List of dictionaries containing data for computation
            id_field: Name of the field to use as the identifier in the result

        Returns:
            Dictionary mapping identifiers to feature values
        """
        results = {}
        for item in batch_data:
            if id_field not in item:
                raise ValueError(f"ID field '{id_field}' not found in data item")

            item_id = item[id_field]
            results[item_id] = self.compute(item)

        return results

    def __eq__(self, other):
        """Check if two features are equal based on name and version."""
        if not isinstance(other, BaseFeature):
            return False
        return self.name == other.name and self.version == other.version

    def __hash__(self):
        """Generate hash based on name and version."""
        return hash((self.name, self.version))


class CachedFeature(BaseFeature):
    """
    Feature with caching capabilities.

    This class extends BaseFeature with caching to avoid re-computing values.

    Attributes:
        cache_enabled: Whether caching is enabled
    """

    def __init__(
        self,
        name: str,
        version: str,
        metadata: FeatureMetadata,
        dependencies: Optional[List[str]] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize a cached feature.

        Args:
            name: Unique name of the feature
            version: Version of the feature
            metadata: Additional information about the feature
            dependencies: List of feature names this feature depends on
            cache_enabled: Whether caching is enabled
        """
        super().__init__(name, version, metadata, dependencies)
        self.cache_enabled = cache_enabled
        self._cache = {}  # Dictionary to store cached values

    def compute(self, data: Dict[str, Any]) -> Any:
        """
        Compute the feature value with caching.

        If caching is enabled and the data has been previously computed,
        the cached value is returned. Otherwise, the value is computed
        and stored in the cache.

        Args:
            data: Dictionary containing data needed for computation

        Returns:
            Computed feature value
        """
        if not self.cache_enabled:
            return self._compute_uncached(data)

        # Generate a cache key based on the input data
        cache_key = self._generate_cache_key(data)

        # Check if value is in cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute and cache the value
        value = self._compute_uncached(data)
        self._cache[cache_key] = value
        return value

    @abstractmethod
    def _compute_uncached(self, data: Dict[str, Any]) -> Any:
        """
        Compute the feature value without caching.

        This method must be implemented by subclasses.

        Args:
            data: Dictionary containing data needed for computation

        Returns:
            Computed feature value
        """
        pass

    def invalidate_cache(self) -> None:
        """Clear the cache for this feature."""
        self._cache.clear()

    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """
        Generate a cache key based on the input data.

        Args:
            data: Dictionary containing data needed for computation

        Returns:
            String cache key
        """
        # Create a copy of the data with non-serializable objects replaced
        serializable_data = {}

        for key, value in data.items():
            # Skip non-serializable objects like DataFrame
            # For DataFrames, we'll use a hash based on contents
            if str(type(value)).find("polars.dataframe") >= 0:
                # For Polars DataFrame, use a hash of the shape and first row
                try:
                    # Use a simplified representation that's still deterministic for the same data
                    serializable_data[key] = (
                        f"DataFrame(shape={value.shape}, hash={hash(tuple(value.row(0)) if not value.is_empty() else 0)})"
                    )
                except Exception:
                    # If there's an issue, use a simple fallback
                    serializable_data[key] = f"DataFrame(shape={value.shape})"
            elif isinstance(value, (int, float, str, bool, type(None))):
                # Basic types can be serialized as-is
                serializable_data[key] = value
            else:
                # For complex objects, use their string representation
                try:
                    serializable_data[key] = str(value)
                except Exception:
                    serializable_data[key] = f"<Object of type {type(value)}>"

        try:
            # Create a JSON string from the data
            serialized = json.dumps(
                serializable_data, sort_keys=True, default=self._json_serializer
            )
            # Hash the string to create a cache key
            cache_key = hashlib.md5(serialized.encode()).hexdigest()
            return cache_key
        except Exception:
            # If there's an error, use a fallback
            return f"{self.name}_{self.version}_{id(data)}"

    def _json_serializer(self, obj: Any) -> str:
        """
        Custom JSON serializer for objects not serializable by default.

        Args:
            obj: Object to serialize

        Returns:
            String representation of the object
        """
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()

        # Handle Polars DataFrames (should not happen due to preprocessing in _generate_cache_key)
        if str(type(obj)).find("polars.dataframe") >= 0:
            return f"DataFrame(shape={obj.shape})"

        # Use string representation for any other non-serializable types
        try:
            return str(obj)
        except Exception:
            return f"<Object of type {type(obj)}>"
