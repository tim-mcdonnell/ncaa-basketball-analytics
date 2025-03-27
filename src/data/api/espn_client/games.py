"""Game-specific ESPN API client methods."""

from typing import Dict, Any, List, Optional
import logging

from src.data.api.exceptions import ResourceNotFoundError
from src.data.api.metadata import get_last_modified, update_last_modified

logger = logging.getLogger(__name__)


class GamesEndpoint:
    """Games endpoint methods for ESPN API client."""

    async def get_games(
        self,
        date_str: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: int = 100,
        groups: Optional[str] = None,
        incremental: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get games list from the ESPN API.

        Args:
            date_str: Single date string (YYYYMMDD)
            team_id: Filter by team ID
            limit: Maximum number of results
            groups: Filter by conference/division groups
            incremental: Whether to use incremental updates
            start_date: Start date for date range (YYYYMMDD)
            end_date: End date for date range (YYYYMMDD)

        Returns:
            List of game data

        Raises:
            ParseError: If API response has invalid structure
            APIError: If API request fails
        """
        try:
            # Prepare query parameters
            params = {}

            # Handle date filtering
            if start_date and end_date:
                params["dates"] = f"{start_date}-{end_date}"
            elif date_str:
                params["dates"] = date_str

            # Add other filters
            if team_id:
                params["team"] = team_id
            if groups:
                params["groups"] = groups
            if limit:
                params["limit"] = str(limit)

            # Get games from API with enhanced recovery
            response = await self.get_with_enhanced_recovery("/scoreboard", params)

            if "events" not in response:
                logger.warning("No events found in response")
                return []

            # Extract and return games data
            games = response["events"]
            logger.info(f"Retrieved {len(games)} games from ESPN API")

            # Update metadata if incremental
            if incremental:
                # Create a resource ID based on the query parameters
                resource_id = None
                if date_str and team_id:
                    resource_id = f"{date_str}-{team_id}"
                elif date_str:
                    resource_id = f"{date_str}-all"
                elif team_id:
                    resource_id = f"team-{team_id}"

                # Update metadata with resource ID if available
                if resource_id:
                    update_last_modified(
                        "games",
                        resource_id=resource_id,
                        metadata_file=self.metadata_file,
                        metadata_dir=self.metadata_dir,
                    )
                else:
                    # Update general games metadata
                    update_last_modified(
                        "games",
                        metadata_file=self.metadata_file,
                        metadata_dir=self.metadata_dir,
                    )

            return games

        except Exception as e:
            logger.error(f"Failed to get games: {e}")
            raise

    async def get_game(self, game_id: str) -> Dict[str, Any]:
        """
        Get game details from the ESPN API.

        Args:
            game_id: Game ID

        Returns:
            Game data

        Raises:
            ResourceNotFoundError: If game not found
            APIError: If API request fails
        """
        try:
            # Check last update time
            last_update = get_last_modified(
                "games",
                resource_id=game_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Game {game_id} last updated: {last_update}")

            # Get game details from API with enhanced recovery
            endpoint = f"/competitions/{game_id}"
            response = await self.get_with_enhanced_recovery(endpoint)

            if "id" not in response:
                raise ResourceNotFoundError("Game", game_id)

            # Update metadata
            update_last_modified(
                "games",
                resource_id=game_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            return response
        except Exception as e:
            logger.error(f"Failed to get game details for {game_id}: {e}")
            raise
