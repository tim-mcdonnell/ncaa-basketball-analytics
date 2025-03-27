"""Player-specific ESPN API client methods."""

from typing import Dict, Any, List, Optional
import logging

from src.data.api.exceptions import ResourceNotFoundError
from src.data.api.metadata import get_last_modified, update_last_modified

logger = logging.getLogger(__name__)


class PlayersEndpoint:
    """Players endpoint methods for ESPN API client."""

    async def get_team_players(self, team_id: str) -> List[Dict[str, Any]]:
        """
        Get team roster from the ESPN API.

        Args:
            team_id: Team ID

        Returns:
            List of player data

        Raises:
            ResourceNotFoundError: If team not found
            APIError: If API request fails
        """
        try:
            # Check last update time
            last_update = get_last_modified(
                "players",
                resource_id=f"team_{team_id}",
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Team {team_id} roster last updated: {last_update}")

            # Get roster from API with enhanced recovery
            endpoint = f"/teams/{team_id}/roster"
            response = await self.get_with_enhanced_recovery(endpoint)

            if "team" not in response or "athletes" not in response["team"]:
                logger.warning(f"Team {team_id} roster data not found or empty")
                return []

            # Extract player data
            players = response["team"]["athletes"]

            # Update metadata
            update_last_modified(
                "players",
                resource_id=f"team_{team_id}",
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            logger.info(f"Retrieved {len(players)} players for team {team_id}")
            return players
        except Exception as e:
            logger.error(f"Failed to get players for team {team_id}: {e}")
            raise

    async def get_player_stats(
        self, player_id: str, season: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a specific player.

        Args:
            player_id: Player ID
            season: Optional season year (e.g., "2023-24")

        Returns:
            Player statistics data

        Raises:
            ResourceNotFoundError: If player not found
            APIError: If API request fails
        """
        try:
            # Check last update time
            metadata_id = f"{player_id}_stats"
            if season:
                metadata_id = f"{metadata_id}_{season}"

            last_update = get_last_modified(
                "players",
                resource_id=metadata_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Player {player_id} stats last updated: {last_update}")

            # Prepare query parameters
            params = {}
            if season:
                params["season"] = season

            # Get player stats from API with enhanced recovery
            endpoint = f"/athletes/{player_id}/statistics"
            response = await self.get_with_enhanced_recovery(endpoint, params)

            if "player" not in response:
                raise ResourceNotFoundError("Player", player_id)

            # Update metadata
            update_last_modified(
                "players",
                resource_id=metadata_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            return response
        except Exception as e:
            logger.error(f"Failed to get stats for player {player_id}: {e}")
            raise
