from typing import List, Dict, Any, Optional
import logging
import asyncio

from src.data.api.espn_client import AsyncESPNClient
from src.data.api.models.team import Team, TeamList, TeamRecord
from src.data.api.exceptions import APIError, ResourceNotFoundError, ParseError

logger = logging.getLogger(__name__)

async def get_all_teams(
    client: Optional[AsyncESPNClient] = None,
    incremental: bool = False
) -> List[Team]:
    """
    Get all NCAA basketball teams.
    
    Args:
        client: Optional AsyncESPNClient instance
        incremental: Whether to use incremental updates
        
    Returns:
        List of Team objects
        
    Raises:
        APIError: If API request fails
    """
    # Create client if not provided
    should_close = False
    if client is None:
        client = AsyncESPNClient()
        should_close = True
    
    try:
        async with client if should_close else asyncio.nullcontext():
            # Get teams from API
            raw_teams = await client.get_teams()
            
            # Convert to Team objects
            teams = []
            for raw_team in raw_teams:
                try:
                    team = Team(
                        id=raw_team["id"],
                        name=raw_team["name"],
                        abbreviation=raw_team.get("abbreviation", "")
                    )
                    teams.append(team)
                except Exception as e:
                    logger.warning(f"Failed to parse team: {e}")
            
            logger.info(f"Successfully retrieved {len(teams)} teams")
            return teams
    except APIError as e:
        logger.error(f"Failed to get teams: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting teams: {e}")
        raise APIError(f"Unexpected error: {str(e)}")

async def get_team_details(team_id: str, client: Optional[AsyncESPNClient] = None) -> Team:
    """
    Get detailed information for a specific team.
    
    Args:
        team_id: Team ID
        client: Optional AsyncESPNClient instance
        
    Returns:
        Team object with details
        
    Raises:
        ResourceNotFoundError: If team not found
        APIError: If API request fails
    """
    # Create client if not provided
    should_close = False
    if client is None:
        client = AsyncESPNClient()
        should_close = True
    
    try:
        async with client if should_close else asyncio.nullcontext():
            # Get team details from API
            raw_team = await client.get_team_details(team_id)
            
            # Parse record if available
            record_summary = raw_team.get("record", "0-0")
            wins, losses = 0, 0
            
            if "-" in record_summary:
                try:
                    wins_str, losses_str = record_summary.split("-")
                    wins = int(wins_str)
                    losses = int(losses_str)
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse record: {record_summary}")
            
            # Create Team object
            team = Team(
                id=raw_team["id"],
                name=raw_team["name"],
                abbreviation=raw_team.get("abbreviation", ""),
                location=raw_team.get("location"),
                logo=raw_team.get("logo"),
                record=TeamRecord(
                    summary=record_summary,
                    wins=wins,
                    losses=losses
                )
            )
            
            logger.info(f"Successfully retrieved details for team {team_id}: {team.name}")
            return team
    except ResourceNotFoundError:
        logger.error(f"Team {team_id} not found")
        raise
    except APIError as e:
        logger.error(f"Failed to get team details for {team_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting team details for {team_id}: {e}")
        raise APIError(f"Unexpected error: {str(e)}")

async def get_teams_batch(team_ids: List[str], client: Optional[AsyncESPNClient] = None) -> List[Team]:
    """
    Get details for multiple teams concurrently.
    
    Args:
        team_ids: List of team IDs
        client: Optional AsyncESPNClient instance
        
    Returns:
        List of Team objects
        
    Raises:
        APIError: If API request fails
    """
    # Create client if not provided
    should_close = False
    if client is None:
        client = AsyncESPNClient()
        should_close = True
    
    try:
        async with client if should_close else asyncio.nullcontext():
            # Create tasks for getting each team
            tasks = [get_team_details(team_id, client) for team_id in team_ids]
            
            # Get teams concurrently
            teams = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_teams = []
            for i, result in enumerate(teams):
                if isinstance(result, Exception):
                    team_id = team_ids[i]
                    if isinstance(result, ResourceNotFoundError):
                        logger.warning(f"Team {team_id} not found")
                    else:
                        logger.error(f"Failed to get team {team_id}: {result}")
                else:
                    valid_teams.append(result)
            
            logger.info(f"Successfully retrieved {len(valid_teams)} of {len(team_ids)} teams in batch")
            return valid_teams
    except Exception as e:
        logger.error(f"Failed to get teams batch: {e}")
        raise APIError(f"Batch team retrieval failed: {str(e)}") 