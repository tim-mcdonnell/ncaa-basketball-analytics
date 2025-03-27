"""
Raw team repository.

This module defines the repository for raw team data.
"""

import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID


from src.data.storage.db import DatabaseManager
from src.data.storage.models.raw.team import RawTeam
from src.data.storage.repositories.base_repository import BaseRepository


class RawTeamRepository(BaseRepository):
    """
    Repository for raw team data from ESPN API.

    This repository manages the storage and retrieval of raw team data
    in the raw_teams table.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the raw team repository.

        Args:
            db_manager: Optional database manager instance. If None, creates a new one.
        """
        super().__init__(db_manager=db_manager, table_name="raw_teams")

    def get_by_team_id(self, team_id: str) -> Optional[RawTeam]:
        """
        Get raw team data by team_id.

        Args:
            team_id: The team identifier to look up

        Returns:
            RawTeam object if found, None otherwise
        """
        result = self.get_by_id(team_id, id_column="team_id")

        if result is None or len(result) == 0:
            return None

        # Convert Polars DataFrame row to dictionary
        row_data = result.row(0)
        row_dict = {result.columns[i]: row_data[i] for i in range(len(result.columns))}

        # Parse the raw_data JSON string back to a dictionary
        if isinstance(row_dict["raw_data"], str):
            row_dict["raw_data"] = json.loads(row_dict["raw_data"])

        return RawTeam(**row_dict)

    def get_latest_team_ids(self) -> List[str]:
        """
        Get a list of all team_ids in the repository.

        Returns:
            List of team_id strings
        """
        query = """
        WITH ranked_teams AS (
            SELECT
                team_id,
                collected_at,
                ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY collected_at DESC) as rn
            FROM raw_teams
        )
        SELECT team_id
        FROM ranked_teams
        WHERE rn = 1
        """

        result = self.query_to_polars(query)
        return result["team_id"].to_list() if not result.is_empty() else []

    def save(self, team: RawTeam) -> UUID:
        """
        Save a raw team to the repository.

        Args:
            team: RawTeam object to save

        Returns:
            UUID of the saved record
        """
        data = team.model_dump(exclude={"id"} if team.id is None else {})

        # Convert any non-serializable data
        if isinstance(data["collected_at"], datetime):
            data["collected_at"] = data["collected_at"].isoformat()

        # Convert raw_data to JSON string if it's a dict
        if isinstance(data["raw_data"], dict):
            data["raw_data"] = json.dumps(data["raw_data"])

        return self.insert(data)

    def save_many(self, teams: List[RawTeam]) -> int:
        """
        Save multiple raw teams to the repository.

        Args:
            teams: List of RawTeam objects to save

        Returns:
            Number of records saved
        """
        if not teams:
            return 0

        # Convert models to dictionaries
        team_dicts = []
        for team in teams:
            data = team.model_dump(exclude={"id"} if team.id is None else {})

            # Convert any non-serializable data
            if isinstance(data["collected_at"], datetime):
                data["collected_at"] = data["collected_at"].isoformat()

            # Convert raw_data to JSON string if it's a dict
            if isinstance(data["raw_data"], dict):
                data["raw_data"] = json.dumps(data["raw_data"])

            team_dicts.append(data)

        return self.insert_many(team_dicts)

    def get_all_teams(self) -> List[RawTeam]:
        """
        Get all raw teams from the repository.

        Returns:
            List of RawTeam objects
        """
        df = self.get_all()

        if df.is_empty():
            return []

        # Convert DataFrame to list of RawTeam objects
        teams = []
        for i in range(len(df)):
            row_data = df.row(i)
            row_dict = {df.columns[j]: row_data[j] for j in range(len(df.columns))}

            # Parse the raw_data JSON string back to a dictionary
            if isinstance(row_dict["raw_data"], str):
                row_dict["raw_data"] = json.loads(row_dict["raw_data"])

            teams.append(RawTeam(**row_dict))

        return teams

    def delete_by_team_id(self, team_id: str) -> int:
        """
        Delete all raw team records with the given team_id.

        Args:
            team_id: Team identifier to delete

        Returns:
            Number of records deleted
        """
        query = """
        DELETE FROM raw_teams
        WHERE team_id = ?
        """

        result = self.execute_query(query, [team_id])
        return result.fetchone()[0]
