"""
Metadata utilities for managing API data tracking.

This module provides functions to manage metadata for incremental data collection,
tracking when resources were last updated to optimize API calls.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

DEFAULT_METADATA_DIR = os.path.join("data", "metadata")
DEFAULT_METADATA_FILE = "espn_metadata.json"


def ensure_metadata_dir(metadata_dir: str = DEFAULT_METADATA_DIR) -> None:
    """
    Ensure the metadata directory exists.

    Args:
        metadata_dir: Path to metadata directory
    """
    os.makedirs(metadata_dir, exist_ok=True)
    logger.debug(f"Ensured metadata directory exists: {metadata_dir}")


def load_metadata(
    metadata_file: str = DEFAULT_METADATA_FILE, metadata_dir: str = DEFAULT_METADATA_DIR
) -> Dict[str, Any]:
    """
    Load metadata from file.

    Args:
        metadata_file: Metadata filename
        metadata_dir: Directory containing metadata files

    Returns:
        Dictionary of metadata
    """
    ensure_metadata_dir(metadata_dir)
    file_path = os.path.join(metadata_dir, metadata_file)

    if not os.path.exists(file_path):
        logger.info(f"Metadata file not found, creating new one: {file_path}")
        return {}

    try:
        with open(file_path, "r") as f:
            metadata = json.load(f)
        logger.debug(f"Loaded metadata from {file_path}")
        return metadata
    except json.JSONDecodeError:
        logger.warning(f"Error decoding metadata file {file_path}, returning empty metadata")
        return {}
    except Exception as e:
        logger.error(f"Error loading metadata from {file_path}: {e}")
        return {}


def save_metadata(
    metadata: Dict[str, Any],
    metadata_file: str = DEFAULT_METADATA_FILE,
    metadata_dir: str = DEFAULT_METADATA_DIR,
) -> None:
    """
    Save metadata to file.

    Args:
        metadata: Dictionary of metadata
        metadata_file: Metadata filename
        metadata_dir: Directory to save metadata file
    """
    ensure_metadata_dir(metadata_dir)
    file_path = os.path.join(metadata_dir, metadata_file)

    try:
        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metadata to {file_path}: {e}")


def update_last_modified(
    resource_type: str,
    resource_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    metadata_file: str = DEFAULT_METADATA_FILE,
    metadata_dir: str = DEFAULT_METADATA_DIR,
) -> None:
    """
    Update the last modified timestamp for a resource.

    Args:
        resource_type: Type of resource (e.g., 'teams', 'games')
        resource_id: Optional ID of specific resource
        timestamp: ISO format timestamp (default: current time)
        metadata_file: Metadata filename
        metadata_dir: Directory containing metadata files
    """
    if timestamp is None:
        # Use timezone-aware datetime (recommended in Python 3.11+)
        timestamp = datetime.now(UTC).isoformat()

    metadata = load_metadata(metadata_file, metadata_dir)

    # Initialize the resource type if it doesn't exist
    if resource_type not in metadata:
        metadata[resource_type] = {}

    # Update the timestamp for the specific resource or the entire type
    if resource_id:
        if "resources" not in metadata[resource_type]:
            metadata[resource_type]["resources"] = {}
        metadata[resource_type]["resources"][resource_id] = timestamp
    else:
        metadata[resource_type]["last_updated"] = timestamp

    # Save the updated metadata
    save_metadata(metadata, metadata_file, metadata_dir)
    logger.debug(
        f"Updated last modified for {resource_type}"
        + (f" resource {resource_id}" if resource_id else "")
    )


def get_last_modified(
    resource_type: str,
    resource_id: Optional[str] = None,
    metadata_file: str = DEFAULT_METADATA_FILE,
    metadata_dir: str = DEFAULT_METADATA_DIR,
) -> Optional[str]:
    """
    Get the last modified timestamp for a resource.

    Args:
        resource_type: Type of resource (e.g., 'teams', 'games')
        resource_id: Optional ID of specific resource
        metadata_file: Metadata filename
        metadata_dir: Directory containing metadata files

    Returns:
        ISO format timestamp or None if not found
    """
    metadata = load_metadata(metadata_file, metadata_dir)

    if resource_type not in metadata:
        return None

    if resource_id:
        if "resources" not in metadata[resource_type]:
            return None
        return metadata[resource_type]["resources"].get(resource_id)

    return metadata[resource_type].get("last_updated")
