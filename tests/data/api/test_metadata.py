import pytest
import os
import json
import tempfile
import shutil
from datetime import datetime, UTC
from unittest.mock import patch

from src.data.api.metadata import (
    ensure_metadata_dir,
    load_metadata,
    save_metadata,
    update_last_modified,
    get_last_modified,
)


class TestMetadata:
    """Test metadata module for incremental data tracking."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing metadata storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after test
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def metadata_file(self):
        """Return a standard metadata filename for tests."""
        return "test_metadata.json"

    def test_ensure_metadata_dir(self, temp_dir):
        """Test that ensure_metadata_dir creates the directory."""
        test_dir = os.path.join(temp_dir, "metadata_test")

        # Directory shouldn't exist yet
        assert not os.path.exists(test_dir)

        # Call function
        ensure_metadata_dir(test_dir)

        # Directory should now exist
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)

        # Should work when called again on existing directory
        ensure_metadata_dir(test_dir)
        assert os.path.exists(test_dir)

    def test_load_metadata_nonexistent_file(self, temp_dir, metadata_file):
        """Test loading metadata when file doesn't exist."""
        metadata_path = os.path.join(temp_dir, metadata_file)

        # File shouldn't exist yet
        assert not os.path.exists(metadata_path)

        # Load should return empty dict
        result = load_metadata(metadata_file, temp_dir)
        assert result == {}

        # Directory should have been created
        assert os.path.exists(temp_dir)

    def test_save_and_load_metadata(self, temp_dir, metadata_file):
        """Test saving and loading metadata."""
        test_metadata = {
            "teams": {
                "last_updated": "2023-01-01T00:00:00",
                "resources": {"123": "2023-01-02T00:00:00"},
            }
        }

        # Save metadata
        save_metadata(test_metadata, metadata_file, temp_dir)

        # File should now exist
        metadata_path = os.path.join(temp_dir, metadata_file)
        assert os.path.exists(metadata_path)

        # Load metadata and verify it matches
        loaded = load_metadata(metadata_file, temp_dir)
        assert loaded == test_metadata

        # Check raw file contents to verify JSON structure
        with open(metadata_path, "r") as f:
            raw_data = json.load(f)
        assert raw_data == test_metadata

    def test_update_last_modified_new_resource(self, temp_dir, metadata_file):
        """Test updating last modified timestamp for a new resource type."""
        # Set a fixed timestamp for testing
        timestamp = "2023-01-01T12:00:00"

        # Update with specific timestamp
        update_last_modified(
            "teams", timestamp=timestamp, metadata_file=metadata_file, metadata_dir=temp_dir
        )

        # Load and verify
        metadata = load_metadata(metadata_file, temp_dir)
        assert "teams" in metadata
        assert metadata["teams"]["last_updated"] == timestamp

    def test_update_last_modified_specific_resource(self, temp_dir, metadata_file):
        """Test updating last modified timestamp for a specific resource."""
        # Set a fixed timestamp for testing
        timestamp = "2023-01-01T12:00:00"
        resource_id = "team123"

        # Update with specific timestamp and resource ID
        update_last_modified(
            "teams",
            resource_id=resource_id,
            timestamp=timestamp,
            metadata_file=metadata_file,
            metadata_dir=temp_dir,
        )

        # Load and verify
        metadata = load_metadata(metadata_file, temp_dir)
        assert "teams" in metadata
        assert "resources" in metadata["teams"]
        assert metadata["teams"]["resources"][resource_id] == timestamp

    def test_update_last_modified_auto_timestamp(self, temp_dir, metadata_file):
        """Test update_last_modified with automatic timestamp generation."""
        with patch("src.data.api.metadata.datetime") as mock_datetime:
            # Mock now() to return a fixed time
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            # Also mock UTC constant
            mock_datetime.UTC = UTC

            # Call without timestamp
            update_last_modified("teams", metadata_file=metadata_file, metadata_dir=temp_dir)

            # Verify timestamp was generated correctly
            metadata = load_metadata(metadata_file, temp_dir)
            assert metadata["teams"]["last_updated"] == mock_now.isoformat()

            # Verify datetime.now was called with UTC
            mock_datetime.now.assert_called_once_with(UTC)

    def test_get_last_modified_nonexistent(self, temp_dir, metadata_file):
        """Test getting last modified for non-existent resource."""
        # Should return None for non-existent resource
        result = get_last_modified(
            "nonexistent", metadata_file=metadata_file, metadata_dir=temp_dir
        )
        assert result is None

    def test_get_last_modified_resource_type(self, temp_dir, metadata_file):
        """Test getting last modified for resource type."""
        # Set up test data
        timestamp = "2023-01-01T12:00:00"
        test_metadata = {"teams": {"last_updated": timestamp}}
        save_metadata(test_metadata, metadata_file, temp_dir)

        # Get timestamp
        result = get_last_modified("teams", metadata_file=metadata_file, metadata_dir=temp_dir)
        assert result == timestamp

    def test_get_last_modified_specific_resource(self, temp_dir, metadata_file):
        """Test getting last modified for specific resource."""
        # Set up test data
        timestamp = "2023-01-01T12:00:00"
        resource_id = "team123"
        test_metadata = {"teams": {"resources": {resource_id: timestamp}}}
        save_metadata(test_metadata, metadata_file, temp_dir)

        # Get timestamp
        result = get_last_modified(
            "teams", resource_id=resource_id, metadata_file=metadata_file, metadata_dir=temp_dir
        )
        assert result == timestamp
