"""
DEPRECATED: This file is deprecated and will be removed in a future release.

Please use the refactored implementation:
    from src.data.api.espn_client.client import AsyncESPNClient, ESPNClient

See the ESPN_API_REFACTORING.md file for more details on the changes.
"""

import warnings
from typing import Dict, Any, List, Optional
import logging
import os
import aiohttp
import asyncio
import json
from urllib.parse import urljoin

from src.data.api.async_client import AsyncClient
from src.data.api.rate_limiter import AdaptiveRateLimiter
from src.data.api.exceptions import (
    APIError,
    RateLimitError,
    ResourceNotFoundError,
    ParseError,
    ValidationError,
    AuthenticationError,
    ServiceUnavailableError,
)
from src.data.api.metadata import get_last_modified, update_last_modified
from src.config import APIConfig
from src.data.api.endpoints.common import get_async_context
from src.data.api.espn_client.client import AsyncESPNClient, ESPNClient

warnings.warn(
    "The espn_client.py file is deprecated and will be removed in a future release. "
    "Please use 'from src.data.api.espn_client.client import AsyncESPNClient, ESPNClient' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AsyncESPNClient", "ESPNClient"]

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_HEADERS = {
    "User-Agent": "NCAA Basketball Analytics Project (tsteve.github@gmail.com)",
    "Accept": "application/json",
}


# Keeping the rest of the file for now, but it will be removed in a future update
# This class is deprecated and provided only for backward compatibility
class DeprecatedAsyncESPNClient(AsyncClient):
    """
    DEPRECATED: Asynchronous ESPN API client specifically for NCAA basketball data.

    This class is deprecated and will be removed in a future release.
    Use src.data.api.espn_client.client.AsyncESPNClient instead.
    """

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

    def __init__(
        self,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
        retry_factor: float = 2.0,
        timeout: float = 30.0,
        rate_limit_initial: int = 10,
        rate_limit_min: int = 1,
        rate_limit_max: int = 50,
        metadata_dir: str = os.path.join("data", "metadata"),
        metadata_file: str = "espn_metadata.json",
        config: Optional[APIConfig] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize ESPN API client.

        Args:
            base_url: Base URL for API requests (default: ESPN college basketball API)
            max_retries: Maximum number of retry attempts
            retry_min_wait: Minimum wait time between retries in seconds
            retry_max_wait: Maximum wait time between retries in seconds
            retry_factor: Exponential factor for retry backoff
            timeout: Request timeout in seconds
            rate_limit_initial: Initial concurrency limit
            rate_limit_min: Minimum concurrency limit
            rate_limit_max: Maximum concurrency limit
            metadata_dir: Directory for metadata storage
            metadata_file: Filename for metadata storage
            config: API configuration
            session: Optional aiohttp session (if not provided, one will be created)
        """
        super().__init__(
            base_url=base_url or self.BASE_URL,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            retry_max_wait=retry_max_wait,
            retry_factor=retry_factor,
            timeout=timeout,
        )
        self.rate_limiter = AdaptiveRateLimiter(
            initial=rate_limit_initial, min_limit=rate_limit_min, max_limit=rate_limit_max
        )
        self.metadata_dir = metadata_dir
        self.metadata_file = metadata_file
        logger.debug(
            f"Initialized AsyncESPNClient with metadata at {os.path.join(metadata_dir, metadata_file)}"
        )
        self.config = config or APIConfig()
        self.sport = self.config.sport
        self.league = self.config.league

        # Session management
        self._session = session
        self._owned_session = False

        # Logging
        logger.debug(f"Initialized AsyncESPNClient with base URL: {self.base_url}")

    async def __aenter__(self):
        """Enter async context manager."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self._owned_session and self._session is not None:
            await self._session.close()
            self._session = None
            self._owned_session = False

    async def _ensure_session(self):
        """Ensure a session exists, creating one if needed."""
        if self._session is None:
            self._session = aiohttp.ClientSession(headers=DEFAULT_HEADERS)
            self._owned_session = True
        return self._session

    async def close(self):
        """Close the session if owned by this client."""
        if self._owned_session and self._session is not None:
            await self._session.close()
            self._session = None
            self._owned_session = False

    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL for the specified endpoint.

        Args:
            endpoint: API endpoint path

        Returns:
            Full URL
        """
        # Remove leading slash if present
        endpoint = endpoint.lstrip("/")
        return urljoin(self.base_url, endpoint)

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make request to ESPN API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            timeout: Request timeout

        Returns:
            JSON response

        Raises:
            APIError: If API request fails
            ResourceNotFoundError: If resource not found
        """
        await self._ensure_session()
        timeout = timeout or DEFAULT_TIMEOUT
        url = self._build_url(endpoint)

        try:
            async with self._session.request(
                method, url, params=params, timeout=timeout
            ) as response:
                # Check for successful response
                if response.status == 404:
                    raise ResourceNotFoundError("Resource", endpoint)

                if response.status != 200:
                    error_msg = f"API request failed with status {response.status}"
                    try:
                        error_data = await response.json()
                        if "message" in error_data:
                            error_msg = f"{error_msg}: {error_data['message']}"
                    except json.JSONDecodeError:
                        # If we can't parse JSON, use text
                        try:
                            error_text = await response.text()
                            error_msg = f"{error_msg}: {error_text[:100]}"
                        except Exception as e:
                            logger.debug(f"Could not get error text: {e}")

                    raise APIError(error_msg)

                # Parse JSON response
                try:
                    data = await response.json()
                    return data
                except json.JSONDecodeError as e:
                    # Log response text for debugging
                    text = await response.text()
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Response text: {text[:200]}...")
                    raise APIError(f"Invalid JSON response: {str(e)}")

        except ResourceNotFoundError:
            # Re-raise without wrapping
            raise
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during request to {url}: {e}")
            raise APIError(f"HTTP error: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"Request to {url} timed out after {timeout}s")
            raise APIError(f"Request timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Unexpected error during request to {url}: {e}")
            raise APIError(f"Unexpected error: {str(e)}")

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
            response = await self._request("GET", path, params=params)
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

    async def get_all_teams(self, incremental: bool = False) -> List[Dict[str, Any]]:
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

            response = await self._request("GET", f"{self.sport}/{self.league}/teams")
            if "sports" not in response or not response["sports"]:
                raise ParseError("Invalid teams response structure")

            teams_data = []
            for sport in response["sports"]:
                for league in sport.get("leagues", []):
                    teams_data.extend(league.get("teams", []))

            # Get the actual team info from the teams data
            teams = [team.get("team", {}) for team in teams_data if "team" in team]

            # Update metadata
            update_last_modified(
                "teams",
                timestamp=None,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            logger.info(f"Retrieved {len(teams)} teams from ESPN API")
            return teams
        except Exception as e:
            logger.error(f"Failed to get teams: {e}")
            raise APIError(f"Failed to get teams: {str(e)}")

    async def get_team(self, team_id: str) -> Dict[str, Any]:
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
            last_update = get_last_modified(
                "teams",
                resource_id=team_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Team {team_id} last updated: {last_update}")

            response = await self._request("GET", f"{self.sport}/{self.league}/teams/{team_id}")
            if "team" not in response:
                raise ResourceNotFoundError("Team", team_id)

            # Update metadata for this specific team
            update_last_modified(
                "teams",
                resource_id=team_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            # Process the team data to extract record summary if available
            team_data = response["team"]
            if (
                "record" in team_data
                and "items" in team_data["record"]
                and team_data["record"]["items"]
            ):
                team_data["record"] = team_data["record"]["items"][0]["summary"]

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
            params = {"limit": str(limit)}

            # Handle date parameters
            if start_date and end_date:
                # Use date range format
                params["dates"] = f"{start_date}-{end_date}"
            elif date_str:
                # Use single date format
                params["dates"] = date_str

            if team_id:
                params["team"] = team_id
            if groups:
                params["groups"] = groups

            # Check if we need to fetch based on incremental flag
            if incremental:
                date_param = params.get("dates", "all")
                resource_id = f"{date_param}-{team_id or 'all'}-{groups or 'all'}"
                last_update = get_last_modified(
                    "games",
                    resource_id=resource_id,
                    metadata_file=self.metadata_file,
                    metadata_dir=self.metadata_dir,
                )
                if last_update:
                    logger.info(
                        f"Using cached games data for {resource_id} (last updated: {last_update})"
                    )
                    return []  # In a real impl, would return cached data

            response = await self._request(
                "GET", f"{self.sport}/{self.league}/scoreboard", params=params
            )
            if "events" not in response:
                raise ParseError("Invalid games response structure")

            games = response["events"]

            # Update metadata
            if incremental:
                date_param = params.get("dates", "all")
                resource_id = f"{date_param}-{team_id or 'all'}-{groups or 'all'}"
                update_last_modified(
                    "games",
                    resource_id=resource_id,
                    metadata_file=self.metadata_file,
                    metadata_dir=self.metadata_dir,
                )

            logger.info(f"Retrieved {len(games)} games from ESPN API")
            return games
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
            # Check if we have metadata for this game
            last_update = get_last_modified(
                "games",
                resource_id=game_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Game {game_id} last updated: {last_update}")

            response = await self._request(
                "GET", f"{self.sport}/{self.league}/summary", params={"event": game_id}
            )

            # Verify the response contains game data
            if "header" not in response or "id" not in response["header"]:
                raise ResourceNotFoundError("Game", game_id)

            # Update metadata for this game
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

    async def get_team_players(self, team_id: str) -> List[Dict[str, Any]]:
        """
        Get players for a team from the ESPN API.

        Args:
            team_id: Team ID

        Returns:
            List of player data

        Raises:
            ResourceNotFoundError: If team not found
            APIError: If API request fails
        """
        try:
            # Check if we have metadata for this team's players
            resource_id = f"{team_id}-players"
            last_update = get_last_modified(
                "players",
                resource_id=resource_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )
            if last_update:
                logger.debug(f"Team {team_id} players last updated: {last_update}")

            response = await self._request(
                "GET", f"{self.sport}/{self.league}/teams/{team_id}/roster"
            )
            if "team" not in response or "athletes" not in response["team"]:
                raise ResourceNotFoundError("Team", team_id)

            players = response["team"]["athletes"]

            # Update metadata for this team's players
            update_last_modified(
                "players",
                resource_id=resource_id,
                metadata_file=self.metadata_file,
                metadata_dir=self.metadata_dir,
            )

            logger.info(f"Retrieved {len(players)} players for team {team_id}")
            return players
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.error(f"Team {team_id} not found")
                raise ResourceNotFoundError("Team", team_id)
            logger.error(f"Failed to get players for team {team_id}: {e}")
            raise APIError(f"Failed to get players for team {team_id}: {str(e)}")
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get players for team {team_id}: {e}")
            raise APIError(f"Failed to get players for team {team_id}: {str(e)}")

    # Add compatibility methods for the existing tests

    async def get_teams(self, incremental: bool = False) -> List[Dict[str, Any]]:
        """
        Get all teams from the ESPN API. This is an alias for get_all_teams for backward compatibility.

        Args:
            incremental: Whether to use incremental updates

        Returns:
            List of team data

        Raises:
            APIError: If API request fails
        """
        return await self.get_all_teams(incremental=incremental)

    async def get_team_details(self, team_id: str) -> Dict[str, Any]:
        """
        Get team details from the ESPN API. This is an alias for get_team for backward compatibility.

        Args:
            team_id: Team ID

        Returns:
            Team data

        Raises:
            ResourceNotFoundError: If team not found
            APIError: If API request fails
        """
        return await self.get_team(team_id)

    def _raise_error_by_status(self, status_code: int, data: Any) -> None:
        """
        Raise appropriate error based on status code.

        Args:
            status_code: HTTP status code
            data: Response data

        Raises:
            Various APIError subclasses
        """
        error_msg = str(data) if not isinstance(data, dict) else data.get("error", str(data))

        if status_code == 400:
            errors = data.get("errors") if isinstance(data, dict) else None
            raise ValidationError(error_msg, errors)
        elif status_code == 401:
            raise AuthenticationError(error_msg)
        elif status_code == 404:
            # Fix constructor call to always provide both parameters
            resource_type = "Resource"
            resource_id = "unknown"
            raise ResourceNotFoundError(resource_type, resource_id, error_msg)
        elif status_code == 429:
            retry_after = None
            if isinstance(data, dict) and "retry_after" in data:
                retry_after = data["retry_after"]
            raise RateLimitError(error_msg, retry_after)
        elif status_code == 503:
            raise ServiceUnavailableError(error_msg)
        else:
            raise APIError(error_msg, status_code)


async def search_teams(
    query: str,
    client: Optional[AsyncESPNClient] = None,
) -> List[Dict[str, Any]]:
    """
    Search for teams by name.

    Args:
        query: Search query
        client: Optional AsyncESPNClient instance

    Returns:
        List of matching teams

    Raises:
        APIError: If API request fails
    """
    # Create client if not provided
    should_close = False
    if client is None:
        client = AsyncESPNClient()
        should_close = True

    try:
        async with get_async_context(client, should_close):
            # Get all teams
            all_teams = await client.get_teams()

            # Filter teams by query (case insensitive)
            query = query.lower()
            matching_teams = [
                team
                for team in all_teams
                if query in team.get("displayName", "").lower()
                or query in team.get("name", "").lower()
                or query in team.get("nickname", "").lower()
                or query in team.get("location", "").lower()
            ]

            logger.info(f"Found {len(matching_teams)} teams matching '{query}'")
            return matching_teams

    except APIError:
        # Re-raise without wrapping
        raise
    except Exception as e:
        logger.error(f"Failed to search teams for '{query}': {e}")
        raise APIError(f"Team search failed: {str(e)}")
