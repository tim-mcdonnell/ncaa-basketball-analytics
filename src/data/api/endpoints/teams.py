from typing import List, Dict, Any, Optional
import logging
import asyncio

from src.data.api.espn_client import AsyncESPNClient
from src.data.api.models.team import Team, TeamList, TeamRecord

logger = logging.getLogger(__name__)

async def get_all_teams(client: Optional[AsyncESPNClient] = None) -> List[Team]:
    """
    Get all NCAA basketball teams.
    
    Args:
        client: Optional AsyncESPNClient instance
        
    Returns:
        List of Team objects
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
            
            return teams
    except Exception as e:
        logger.error(f"Failed to get teams: {e}")
        raise

async def get_team_details(team_id: str, client: Optional[AsyncESPNClient] = None) -> Team:
    """
    Get detailed information for a specific team.
    
    Args:
        team_id: Team ID
        client: Optional AsyncESPNClient instance
        
    Returns:
        Team object with details
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
            
            return team
    except Exception as e:
        logger.error(f"Failed to get team details for {team_id}: {e}")
        raise

async def get_teams_batch(team_ids: List[str], client: Optional[AsyncESPNClient] = None) -> List[Team]:
    """
    Get details for multiple teams concurrently.
    
    Args:
        team_ids: List of team IDs
        client: Optional AsyncESPNClient instance
        
    Returns:
        List of Team objects
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
            for i, team in enumerate(teams):
                if isinstance(team, Exception):
                    logger.error(f"Failed to get team {team_ids[i]}: {team}")
                else:
                    valid_teams.append(team)
            
            return valid_teams
    except Exception as e:
        logger.error(f"Failed to get teams batch: {e}")
        raise 