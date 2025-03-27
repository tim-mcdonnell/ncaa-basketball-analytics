"""
Feature versioning module.

This module provides utilities for managing feature versions.
"""

from typing import Tuple, Optional
import re
from packaging import version


def parse_version_string(version_str: str) -> version.Version:
    """
    Parse a version string into a comparable version object.

    Args:
        version_str: Version string in the format "X.Y.Z"

    Returns:
        Version object that can be compared
    """
    return version.parse(version_str)


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.

    Args:
        version1: First version string
        version2: Second version string

    Returns:
        -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2
    """
    v1 = parse_version_string(version1)
    v2 = parse_version_string(version2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def is_newer_version(version1: str, version2: str) -> bool:
    """
    Check if version1 is newer than version2.

    Args:
        version1: First version string
        version2: Second version string

    Returns:
        True if version1 is newer than version2, False otherwise
    """
    return compare_versions(version1, version2) > 0


def parse_feature_key(feature_key: str) -> Tuple[str, Optional[str]]:
    """
    Parse a feature key into name and version components.

    Args:
        feature_key: Feature key in the format "name@version" or just "name"

    Returns:
        Tuple of (name, version) where version may be None
    """
    match = re.match(r"([^@]+)(?:@(.+))?", feature_key)
    if not match:
        return feature_key, None

    name, version = match.groups()
    return name, version


def format_feature_key(name: str, version: Optional[str] = None) -> str:
    """
    Format a feature name and version into a feature key.

    Args:
        name: Feature name
        version: Optional feature version

    Returns:
        Feature key in the format "name@version" or just "name" if version is None
    """
    if version:
        return f"{name}@{version}"
    return name
