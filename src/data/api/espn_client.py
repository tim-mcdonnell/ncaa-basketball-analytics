from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from src.data.api.async_client import AsyncClient
from src.data.api.rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)

class AsyncESPNClient(AsyncClient):
    """
    Asynchronous ESPN API client specifically for NCAA basketball data.
    
    This client extends the base AsyncClient with ESPN-specific endpoints
    and data processing for college basketball data.
    """
    
    def __init__(
        self,
        timeout: int = 60,
        api_key: Optional[str] = None,
        rate_limiter: Optional[AdaptiveRateLimiter] = None
    ):
        """
        Initialize the ESPN API client.
        
        Args:
            timeout: Request timeout in seconds
            api_key: Optional ESPN API key
            rate_limiter: Optional custom rate limiter
        """
        # ESPN API base URL for men's college basketball
        base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        
        # Set up headers with API key if provided
        headers = {}
        if api_key:
            headers["X-API-KEY"] = api_key
        
        super().__init__(base_url=base_url, timeout=timeout, headers=headers)
        
        # Set up rate limiter with default values if not provided
        self.rate_limiter = rate_limiter or AdaptiveRateLimiter(
            initial=5,  # Start conservatively
            min_limit=1,
            max_limit=20
        )
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a rate-limited GET request to the ESPN API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            headers: Additional headers for the request
            
        Returns:
            JSON response data
        """
        try:
            # Acquire permission from rate limiter
            await self.rate_limiter.acquire()
            
            # Make the request
            result = await super().get(endpoint, params, headers)
            
            # Release the rate limiter with success
            await self.rate_limiter.release(success=True)
            
            return result
        except Exception as e:
            # Release the rate limiter with failure
            await self.rate_limiter.release(success=False)
            logger.error(f"Error making request to {endpoint}: {str(e)}")
            raise
    
    async def get_teams(self) -> List[Dict[str, Any]]:
        """
        Get list of all NCAA basketball teams.
        
        Returns:
            List of team data dictionaries with id, name, and abbreviation
        """
        response = await self.get("/teams")
        
        # Process the response to extract team information
        teams = []
        try:
            for sport in response.get("sports", []):
                for league in sport.get("leagues", []):
                    for team_item in league.get("teams", []):
                        team_data = team_item.get("team", {})
                        teams.append({
                            "id": team_data.get("id"),
                            "name": team_data.get("name"),
                            "abbreviation": team_data.get("abbreviation", "")
                        })
        except (KeyError, TypeError) as e:
            logger.error(f"Error processing teams response: {str(e)}")
            raise ValueError(f"Invalid team data format: {str(e)}")
        
        return teams
    
    async def get_team_details(self, team_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific team.
        
        Args:
            team_id: ESPN team ID
            
        Returns:
            Team details dictionary
        """
        response = await self.get(f"/teams/{team_id}")
        
        # Process the response to extract team details
        try:
            team_data = response.get("team", {})
            record_summary = "0-0"  # Default if no record found
            
            # Extract record if available
            if "record" in team_data:
                record_items = team_data.get("record", {}).get("items", [])
                if record_items:
                    record_summary = record_items[0].get("summary", "0-0")
            
            return {
                "id": team_data.get("id"),
                "name": team_data.get("name"),
                "abbreviation": team_data.get("abbreviation", ""),
                "logo": team_data.get("logo", ""),
                "location": team_data.get("location", ""),
                "record": record_summary
            }
        except (KeyError, TypeError) as e:
            logger.error(f"Error processing team details response: {str(e)}")
            raise ValueError(f"Invalid team details format: {str(e)}")
    
    async def get_games(
        self,
        start_date: str,
        end_date: str,
        team_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get games within a date range, optionally filtered by team.
        
        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            team_id: Optional team ID to filter games
            limit: Maximum number of games to return
            
        Returns:
            List of game data dictionaries
        """
        params = {
            "dates": f"{start_date}-{end_date}",
            "limit": limit
        }
        
        if team_id:
            params["team"] = team_id
        
        response = await self.get("/scoreboard", params=params)
        
        # Process the response to extract game information
        games = []
        try:
            for event in response.get("events", []):
                game_data = {
                    "id": event.get("id"),
                    "date": event.get("date"),
                    "name": event.get("name"),
                    "short_name": event.get("shortName", ""),
                    "status": "scheduled"
                }
                
                # Extract competition details
                competitions = event.get("competitions", [])
                if competitions:
                    competition = competitions[0]
                    
                    # Extract status
                    status = competition.get("status", {}).get("type", {})
                    if status.get("completed", False):
                        game_data["status"] = "completed"
                    elif status.get("state", "") == "in":
                        game_data["status"] = "in_progress"
                    
                    # Extract team and score information
                    competitors = competition.get("competitors", [])
                    game_data["teams"] = []
                    
                    for competitor in competitors:
                        team_info = competitor.get("team", {})
                        team = {
                            "id": team_info.get("id"),
                            "name": team_info.get("name", ""),
                            "score": competitor.get("score", "0")
                        }
                        game_data["teams"].append(team)
                
                games.append(game_data)
        except (KeyError, TypeError) as e:
            logger.error(f"Error processing games response: {str(e)}")
            raise ValueError(f"Invalid games data format: {str(e)}")
        
        return games
    
    async def get_team_players(self, team_id: str) -> List[Dict[str, Any]]:
        """
        Get roster of players for a specific team.
        
        Args:
            team_id: ESPN team ID
            
        Returns:
            List of player data dictionaries
        """
        response = await self.get(f"/teams/{team_id}/roster")
        
        # Process the response to extract player information
        players = []
        try:
            for athlete in response.get("athletes", []):
                player = {
                    "id": athlete.get("id"),
                    "name": athlete.get("fullName"),
                    "jersey": athlete.get("jersey", ""),
                    "position": athlete.get("position", {}).get("name", "Unknown"),
                    "height": athlete.get("height", ""),
                    "weight": athlete.get("weight", ""),
                    "year": athlete.get("class", {}).get("name", "")
                }
                players.append(player)
        except (KeyError, TypeError) as e:
            logger.error(f"Error processing players response: {str(e)}")
            raise ValueError(f"Invalid player data format: {str(e)}")
        
        return players 