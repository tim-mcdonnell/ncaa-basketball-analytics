"""Team-specific ESPN API client methods."""

from typing import Dict, Any, List
import logging

from src.data.api.exceptions import ParseError, ResourceNotFoundError
from src.data.api.metadata import get_last_modified, update_last_modified

logger = logging.getLogger(__name__)


class TeamsEndpoint:
    """Teams endpoint methods for ESPN API client."""

    async def get_teams(self, incremental: bool = False) -> List[Dict[str, Any]]:
        """
        Get all teams from the ESPN API.

        Args:
            incremental: Whether to use incremental updates

        Returns:
            List of team data

        Raises:
            ParseError: If API response has invalid structure
            APIError: If API request fails
        """
        try:
            # Check if we need to fetch
            if incremental:
                last_update = get_last_modified(
                    "teams", metadata_file=self.metadata_file, metadata_dir=self.metadata_dir
                )
                if last_update:
                    logger.info(f"Using cached teams data (last updated: {last_update})")
                    return []  # In a real impl, would return cached data

            # Get teams from API
            response = await self.get("/teams")
            if "sports" not in response or not response["sports"]:
                raise ParseError("Invalid teams response structure")

            # Extract teams data
            teams_data = []
            for sport in response["sports"]:
                for league in sport.get("leagues", []):
                    teams_data.extend(league.get("teams", []))

            # Get the actual team info
            teams = [team.get("team", {}) for team in teams_data if "team" in team]

            # Update metadata
            update_last_modified(
                "teams", metadata_file=self.metadata_file, metadata_dir=self.metadata_dir
            )

            logger.info(f"Retrieved {len(teams)} teams from ESPN API")
            return teams
        except Exception as e:
            logger.error(f"Failed to get teams: {e}")
            raise

    async def get_team_details(self, team_id: str) -> Dict[str, Any]:
        """
        Get team details from the ESPN API.

        Args:
            team_id: Team ID

        Returns:
            Team data

        Raises:
            ResourceNotFoundError: If team not found
            APIError: If API request fails
        """
        try:
            # Check last update time
            last_update = get_last_modified(
                "teams",
                resource_id=team_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Team {team_id} last updated: {last_update}")

            # Get team details from API
            response = await self.get_with_enhanced_recovery(f"/teams/{team_id}")
            if "team" not in response:
                raise ResourceNotFoundError("Team", team_id)

            # Extract team data
            team_data = response["team"]

            # Process record data
            if (
                "record" in team_data
                and isinstance(team_data["record"], dict)
                and "items" in team_data["record"]
                and team_data["record"]["items"]
            ):
                team_data["record"] = team_data["record"]["items"][0]["summary"]

            # Update metadata
            update_last_modified(
                "teams",
                resource_id=team_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            return team_data
        except Exception as e:
            logger.error(f"Failed to get team details for {team_id}: {e}")
            raise
