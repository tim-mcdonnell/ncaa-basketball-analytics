"""ESPN API client implementation."""

from typing import Dict, Any, List, Optional
import logging
import os
import asyncio
import aiohttp

from src.data.api.async_client import AsyncClient
from src.data.api.rate_limiter import AdaptiveRateLimiter
from src.data.api.espn_client.config import ESPNConfig, load_espn_config
from src.data.api.exceptions import (
    APIError,
    RateLimitError,
    ResourceNotFoundError,
    ParseError,
)
from src.data.api.metadata import get_last_modified, update_last_modified

logger = logging.getLogger(__name__)


class AsyncESPNClient(AsyncClient):
    """
    Asynchronous ESPN API client specifically for NCAA basketball data.

    This client extends the base AsyncClient with ESPN-specific endpoints
    and data processing for college basketball data.
    """

    def __init__(
        self,
        config: Optional[ESPNConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize ESPN API client.

        Args:
            config: Preconfigured ESPNConfig object, if provided
            config_path: Path to config file (default from config/api/espn.yaml)
        """
        # Load config if not provided
        self.config = config or load_espn_config(config_path)

        # Initialize base client
        super().__init__(
            base_url=self.config.base_url,
            max_retries=self.config.retries.max_attempts,
            retry_min_wait=self.config.retries.min_wait,
            retry_max_wait=self.config.retries.max_wait,
            retry_factor=self.config.retries.factor,
            timeout=self.config.timeout,
        )

        # Set up rate limiter
        self.rate_limiter = AdaptiveRateLimiter(
            initial=self.config.rate_limiting.initial,
            min_limit=self.config.rate_limiting.min_limit,
            max_limit=self.config.rate_limiting.max_limit,
            success_threshold=self.config.rate_limiting.success_threshold,
            failure_threshold=self.config.rate_limiting.failure_threshold,
        )

        # Initialize metadata paths
        self.metadata_dir = self.config.metadata.dir
        self.metadata_file = self.config.metadata.file

        logger.debug(
            f"Initialized AsyncESPNClient with metadata at {os.path.join(self.metadata_dir, self.metadata_file)}"
        )

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make GET request to ESPN API with rate limiting.

        Args:
            path: API endpoint path
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            APIError: If request fails
            RateLimitError: If rate limit exceeded
        """
        await self.rate_limiter.acquire()
        try:
            response = await super().get(path, params)
            await self.rate_limiter.release(success=True)
            return response
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                await self.rate_limiter.release(success=False)
                raise RateLimitError("ESPN API rate limit exceeded")
            await self.rate_limiter.release(success=False)
            raise APIError(f"ESPN API request failed: {str(e)}")
        except Exception as e:
            await self.rate_limiter.release(success=False)
            raise APIError(f"ESPN API request failed: {str(e)}")

    # Team endpoints
    async def get_teams(self, incremental: bool = False) -> List[Dict[str, Any]]:
        """
        Get all teams from the ESPN API.

        Args:
            incremental: Whether to use incremental updates

        Returns:
            List of team data

        Raises:
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
            raise APIError(f"Failed to get teams: {str(e)}")

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
            response = await self.get(f"/teams/{team_id}")
            if "team" not in response:
                raise ResourceNotFoundError("Team", team_id)

            # Extract team data
            team_data = response["team"]

            # Process record data
            if (
                "record" in team_data
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

            logger.info(f"Retrieved team {team_id} details")
            return team_data
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.error(f"Team {team_id} not found")
                raise ResourceNotFoundError("Team", team_id)
            logger.error(f"Failed to get team {team_id}: {e}")
            raise APIError(f"Failed to get team {team_id}: {str(e)}")
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get team {team_id}: {e}")
            raise APIError(f"Failed to get team {team_id}: {str(e)}")

    # Game endpoints
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
        Get games from the ESPN API.

        Args:
            date_str: Date string (YYYYMMDD)
            team_id: Team ID to filter by
            limit: Maximum number of games to return
            groups: Groups filter (e.g., "50" for top 25 teams)
            incremental: Whether to use incremental updates
            start_date: Start date for date range (YYYYMMDD)
            end_date: End date for date range (YYYYMMDD)

        Returns:
            List of game data

        Raises:
            APIError: If API request fails
        """
        try:
            # Set up parameters
            params = {"limit": str(limit)}

            # Handle date parameters
            if start_date and end_date:
                params["dates"] = f"{start_date}-{end_date}"
            elif date_str:
                params["dates"] = date_str

            # Add optional filters
            if team_id:
                params["team"] = team_id
            if groups:
                params["groups"] = groups

            # Get games from API
            response = await self.get("/scoreboard", params)

            # Extract events
            if "events" not in response:
                return []

            # Validate events
            events = response["events"]
            logger.info(f"Retrieved {len(events)} games from ESPN API")
            return events
        except Exception as e:
            logger.error(f"Failed to get games: {e}")
            raise APIError(f"Failed to get games: {str(e)}")

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
            # Check cache
            last_update = get_last_modified(
                "games",
                resource_id=game_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Game {game_id} last updated: {last_update}")

            # Get game details from API
            response = await self.get(f"/competitions/{game_id}")
            if not response:
                raise ResourceNotFoundError("Game", game_id)

            # Update metadata
            update_last_modified(
                "games",
                resource_id=game_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            logger.info(f"Retrieved game {game_id} details")
            return response
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.error(f"Game {game_id} not found")
                raise ResourceNotFoundError("Game", game_id)
            logger.error(f"Failed to get game {game_id}: {e}")
            raise APIError(f"Failed to get game {game_id}: {str(e)}")
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get game {game_id}: {e}")
            raise APIError(f"Failed to get game {game_id}: {str(e)}")

    # Player endpoints
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
            # Check cache
            last_update = get_last_modified(
                "rosters",
                resource_id=team_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Roster for team {team_id} last updated: {last_update}")

            # Get roster from API
            response = await self.get(f"/teams/{team_id}/roster")
            if "team" not in response or "athletes" not in response["team"]:
                if "team" in response:
                    # Team exists but has no roster
                    return []
                else:
                    # Team doesn't exist
                    raise ResourceNotFoundError("Team", team_id)

            # Extract players
            players = response["team"]["athletes"]

            # Update metadata
            update_last_modified(
                "rosters",
                resource_id=team_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            logger.info(f"Retrieved {len(players)} players for team {team_id}")
            return players
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.error(f"Team {team_id} not found")
                raise ResourceNotFoundError("Team", team_id)
            logger.error(f"Failed to get roster for team {team_id}: {e}")
            raise APIError(f"Failed to get roster for team {team_id}: {str(e)}")
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get roster for team {team_id}: {e}")
            raise APIError(f"Failed to get roster for team {team_id}: {str(e)}")


class ESPNClient:
    """
    Synchronous wrapper for AsyncESPNClient.

    This is a convenience class for non-async code to use the ESPN API client.
    """

    def __init__(
        self,
        config: Optional[ESPNConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize synchronous ESPN API client.

        Args:
            config: Preconfigured ESPNConfig object, if provided
            config_path: Path to config file (default from config/api/espn.yaml)
        """
        self.async_client = AsyncESPNClient(config, config_path)

    def __enter__(self):
        """Context manager entry."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.async_client.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        try:
            self._loop.run_until_complete(self.async_client.__aexit__(exc_type, exc_val, exc_tb))
        finally:
            self._loop.close()

    # Team endpoints
    def get_teams(self, incremental: bool = False) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_teams."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_client.get_teams(incremental))

    def get_team_details(self, team_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for get_team_details."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_client.get_team_details(team_id))

    # Game endpoints
    def get_games(
        self,
        date_str: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: int = 100,
        groups: Optional[str] = None,
        incremental: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_games."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.async_client.get_games(
                date_str, team_id, limit, groups, incremental, start_date, end_date
            )
        )

    def get_game(self, game_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for get_game."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_client.get_game(game_id))

    # Player endpoints
    def get_team_players(self, team_id: str) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_team_players."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_client.get_team_players(team_id))
