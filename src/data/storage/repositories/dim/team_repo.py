"""
Dimensional team repository.

This module defines the repository for dimensional team data.
"""

from datetime import datetime
from typing import Dict, List, Optional


from src.data.storage.db import DatabaseManager
from src.data.storage.models.dim.team import DimTeam
from src.data.storage.repositories.base_repository import BaseRepository


class DimTeamRepository(BaseRepository):
    """
    Repository for dimensional team data.

    This repository manages the storage and retrieval of dimensional team data
    in the dim_teams table.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the dimensional team repository.

        Args:
            db_manager: Optional database manager instance. If None, creates a new one.
        """
        super().__init__(db_manager=db_manager, table_name="dim_teams")

    def get_by_team_id(self, team_id: str) -> Optional[DimTeam]:
        """
        Get dimensional team data by team_id.

        Args:
            team_id: The team identifier to look up

        Returns:
            DimTeam object if found, None otherwise
        """
        result = self.get_by_id(team_id, id_column="team_id")

        if result is None or len(result) == 0:
            return None

        # Convert Polars DataFrame row to dictionary and create DimTeam
        row_dict = result.row(0)
        return DimTeam(**row_dict)

    def get_all_teams(self) -> List[DimTeam]:
        """
        Get all dimensional teams from the repository.

        Returns:
            List of DimTeam objects
        """
        df = self.get_all()

        if df.is_empty():
            return []

        # Convert DataFrame to list of DimTeam objects
        teams = []
        for row in df.iter_rows(named=True):
            teams.append(DimTeam(**row))

        return teams

    def get_teams_by_conference(self, conference: str) -> List[DimTeam]:
        """
        Get teams filtered by conference.

        Args:
            conference: Conference name to filter by

        Returns:
            List of DimTeam objects in the specified conference
        """
        query = """
        SELECT * FROM dim_teams
        WHERE conference = ?
        """

        result = self.query_to_polars(query, [conference])

        if result.is_empty():
            return []

        teams = []
        for row in result.iter_rows(named=True):
            teams.append(DimTeam(**row))

        return teams

    def save(self, team: DimTeam) -> str:
        """
        Save a dimensional team to the repository.

        If the team already exists (by team_id), it updates the record.
        Otherwise, it inserts a new record.

        Args:
            team: DimTeam object to save

        Returns:
            team_id of the saved record
        """
        # Set updated_at to current time
        team.updated_at = datetime.now()

        # Convert model to dictionary
        data = team.model_dump()

        # Convert datetime objects to strings for storage
        if isinstance(data["created_at"], datetime):
            data["created_at"] = data["created_at"].isoformat()
        if isinstance(data["updated_at"], datetime):
            data["updated_at"] = data["updated_at"].isoformat()

        # Check if team exists
        existing_team = self.get_by_team_id(team.team_id)

        if existing_team:
            # If exists, update
            self.update(team.team_id, data, id_column="team_id")
            return team.team_id
        else:
            # If not exists, insert
            return self.insert(data)

    def save_many(self, teams: List[DimTeam]) -> int:
        """
        Save multiple dimensional teams to the repository.

        For each team, if it already exists (by team_id), it updates the record.
        Otherwise, it inserts a new record.

        Args:
            teams: List of DimTeam objects to save

        Returns:
            Number of records saved
        """
        if not teams:
            return 0

        # Process each team individually to handle updates
        saved_count = 0
        for team in teams:
            self.save(team)
            saved_count += 1

        return saved_count

    def delete_by_team_id(self, team_id: str) -> int:
        """
        Delete the dimensional team with the given team_id.

        Args:
            team_id: Team identifier to delete

        Returns:
            Number of records deleted (0 or 1)
        """
        return self.delete(team_id, id_column="team_id")

    def transform_from_raw(self, raw_data: Dict) -> DimTeam:
        """
        Transform raw team data into dimensional team model.

        Args:
            raw_data: Raw team data from the ESPN API

        Returns:
            DimTeam object with transformed data
        """
        # Extract data from the raw JSON
        # This will need to be customized based on the actual ESPN API response structure
        team_id = raw_data.get("id", "")
        name = raw_data.get("displayName", "") or raw_data.get("name", "")

        # Extract other fields
        try:
            conference = raw_data.get("conference", {}).get("name", None)
        except (AttributeError, TypeError):
            conference = None

        try:
            division = raw_data.get("conference", {}).get("division", {}).get("name", None)
        except (AttributeError, TypeError):
            division = None

        try:
            logo_url = raw_data.get("logos", [{}])[0].get("href", None)
        except (IndexError, AttributeError, TypeError):
            logo_url = None

        # Create the dimensional team object
        return DimTeam(
            team_id=team_id,
            name=name,
            conference=conference,
            division=division,
            logo_url=logo_url,
            # Add other fields as needed
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
