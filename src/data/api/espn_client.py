from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timezone
import os
import json
import aiohttp
import asyncio

from src.data.api.async_client import AsyncClient
from src.data.api.rate_limiter import AdaptiveRateLimiter
from src.data.api.exceptions import (
    APIError,
    RateLimitError,
    ResourceNotFoundError,
    ParseError,
    ValidationError,
    AuthenticationError,
    ServiceUnavailableError
)
from src.data.api.metadata import get_last_modified, update_last_modified

logger = logging.getLogger(__name__)

class AsyncESPNClient(AsyncClient):
    """
    Asynchronous ESPN API client specifically for NCAA basketball data.
    
    This client extends the base AsyncClient with ESPN-specific endpoints
    and data processing for college basketball data.
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
        metadata_file: str = "espn_metadata.json"
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
        """
        super().__init__(
            base_url=base_url or self.BASE_URL,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            retry_max_wait=retry_max_wait,
            retry_factor=retry_factor,
            timeout=timeout
        )
        self.rate_limiter = AdaptiveRateLimiter(
            initial=rate_limit_initial,
            min_limit=rate_limit_min,
            max_limit=rate_limit_max
        )
        self.metadata_dir = metadata_dir
        self.metadata_file = metadata_file
        logger.debug(f"Initialized AsyncESPNClient with metadata at {os.path.join(metadata_dir, metadata_file)}")
    
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
                    "teams", 
                    metadata_file=self.metadata_file, 
                    metadata_dir=self.metadata_dir
                )
                if last_update:
                    logger.info(f"Using cached teams data (last updated: {last_update})")
                    return []  # In a real impl, would return cached data

            response = await self.get("/teams")
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
                metadata_dir=self.metadata_dir
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
                metadata_dir=self.metadata_dir
            )
            if last_update:
                logger.debug(f"Team {team_id} last updated: {last_update}")
            
            response = await self.get(f"/teams/{team_id}")
            if "team" not in response:
                raise ResourceNotFoundError("Team", team_id)
            
            # Update metadata for this specific team
            update_last_modified(
                "teams", 
                resource_id=team_id, 
                metadata_file=self.metadata_file, 
                metadata_dir=self.metadata_dir
            )
            
            # Process the team data to extract record summary if available
            team_data = response["team"]
            if "record" in team_data and "items" in team_data["record"] and team_data["record"]["items"]:
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
        end_date: Optional[str] = None
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
                    metadata_dir=self.metadata_dir
                )
                if last_update:
                    logger.info(f"Using cached games data for {resource_id} (last updated: {last_update})")
                    return []  # In a real impl, would return cached data
            
            response = await self.get("/scoreboard", params)
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
                    metadata_dir=self.metadata_dir
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
                metadata_dir=self.metadata_dir
            )
            if last_update:
                logger.debug(f"Game {game_id} last updated: {last_update}")
                
            response = await self.get(f"/summary", {"event": game_id})
            
            # Verify the response contains game data
            if "header" not in response or "id" not in response["header"]:
                raise ResourceNotFoundError("Game", game_id)
            
            # Update metadata for this game
            update_last_modified(
                "games", 
                resource_id=game_id, 
                metadata_file=self.metadata_file, 
                metadata_dir=self.metadata_dir
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
                metadata_dir=self.metadata_dir
            )
            if last_update:
                logger.debug(f"Team {team_id} players last updated: {last_update}")
            
            response = await self.get(f"/teams/{team_id}/roster")
            if "team" not in response or "athletes" not in response["team"]:
                raise ResourceNotFoundError("Team", team_id)
            
            players = response["team"]["athletes"]
            
            # Update metadata for this team's players
            update_last_modified(
                "players", 
                resource_id=resource_id, 
                metadata_file=self.metadata_file, 
                metadata_dir=self.metadata_dir
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