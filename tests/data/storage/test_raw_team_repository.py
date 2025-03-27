"""Tests for the raw team repository."""

import tempfile
import os
from datetime import datetime
from uuid import UUID

import pytest

from src.data.storage.db import DatabaseManager
from src.data.storage.models.raw.team import RawTeam
from src.data.storage.repositories.raw.team_repo import RawTeamRepository


@pytest.fixture
def test_db_manager():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = os.path.join(temp_dir, "test.duckdb")

        db_manager = DatabaseManager(db_path=temp_db_path)
        db_manager.initialize_schema()

        yield db_manager

        # Clean up
        db_manager.close_connection()
        # Cleanup of the directory is handled by the context manager


@pytest.fixture
def sample_raw_team():
    """Create a sample raw team for testing."""
    return RawTeam(
        team_id="MICH",
        raw_data={"name": "Michigan Wolverines", "conference": "Big Ten"},
        source_url="https://api.espn.com/v1/teams/MICH",
        processing_version="1.0",
    )


def test_raw_team_ingestion(test_db_manager, sample_raw_team):
    """Test that raw team data is correctly stored in the repository."""
    # Arrange
    repo = RawTeamRepository(db_manager=test_db_manager)

    # Act
    uuid = repo.save(sample_raw_team)

    # Assert
    assert uuid is not None

    # Retrieve the team and verify it matches
    saved_team = repo.get_by_team_id(sample_raw_team.team_id)
    assert saved_team is not None
    assert saved_team.team_id == sample_raw_team.team_id
    assert saved_team.raw_data == sample_raw_team.raw_data
    assert saved_team.source_url == sample_raw_team.source_url
    assert saved_team.processing_version == sample_raw_team.processing_version

    # Check that the UUID is in the expected format
    assert isinstance(saved_team.id, UUID)


def test_get_latest_team_ids(test_db_manager):
    """Test retrieving the list of latest team IDs."""
    # Arrange
    repo = RawTeamRepository(db_manager=test_db_manager)

    # Create multiple teams with unique IDs
    teams = [
        RawTeam(
            team_id="MICH1",  # Make these unique to avoid constraint violations
            raw_data={"name": "Michigan Wolverines", "conference": "Big Ten"},
            source_url="https://api.espn.com/v1/teams/MICH1",
            processing_version="1.0",
        ),
        RawTeam(
            team_id="OSU",
            raw_data={"name": "Ohio State Buckeyes", "conference": "Big Ten"},
            source_url="https://api.espn.com/v1/teams/OSU",
            processing_version="1.0",
        ),
        # Add a second version of MICH1 with later timestamp
        RawTeam(
            team_id="MICH2",  # Make this unique too
            raw_data={"name": "Michigan Wolverines Updated", "conference": "Big Ten"},
            source_url="https://api.espn.com/v1/teams/MICH2",
            processing_version="1.1",
            collected_at=datetime.now(),  # This will be later than the first one
        ),
    ]

    # Act - save each team
    for team in teams:
        repo.save(team)

    # Assert
    team_ids = repo.get_latest_team_ids()
    assert len(team_ids) == 3
    assert "MICH1" in team_ids
    assert "MICH2" in team_ids
    assert "OSU" in team_ids


def test_save_many(test_db_manager):
    """Test saving multiple teams at once."""
    # Arrange
    repo = RawTeamRepository(db_manager=test_db_manager)

    teams = [
        RawTeam(
            team_id="MICH",
            raw_data={"name": "Michigan Wolverines", "conference": "Big Ten"},
            source_url="https://api.espn.com/v1/teams/MICH",
            processing_version="1.0",
        ),
        RawTeam(
            team_id="OSU",
            raw_data={"name": "Ohio State Buckeyes", "conference": "Big Ten"},
            source_url="https://api.espn.com/v1/teams/OSU",
            processing_version="1.0",
        ),
        RawTeam(
            team_id="PSU",
            raw_data={"name": "Penn State Nittany Lions", "conference": "Big Ten"},
            source_url="https://api.espn.com/v1/teams/PSU",
            processing_version="1.0",
        ),
    ]

    # Act
    num_saved = repo.save_many(teams)

    # Assert
    assert num_saved == 3

    # Verify all teams were saved
    all_teams = repo.get_all_teams()
    assert len(all_teams) == 3
    team_ids = [team.team_id for team in all_teams]
    assert "MICH" in team_ids
    assert "OSU" in team_ids
    assert "PSU" in team_ids


def test_delete_by_team_id(test_db_manager, sample_raw_team):
    """Test deleting a team by team_id."""
    # Arrange
    repo = RawTeamRepository(db_manager=test_db_manager)
    repo.save(sample_raw_team)

    # Confirm the team exists
    team = repo.get_by_team_id(sample_raw_team.team_id)
    assert team is not None

    # Act
    num_deleted = repo.delete_by_team_id(sample_raw_team.team_id)

    # Assert
    assert num_deleted == 1

    # Verify the team was deleted
    team = repo.get_by_team_id(sample_raw_team.team_id)
    assert team is None
