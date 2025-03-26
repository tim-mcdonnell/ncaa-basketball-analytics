from typing import List, Dict, Any, Optional
import logging
import asyncio

from src.data.api.espn_client import AsyncESPNClient
from src.data.api.models.player import Player, PlayerList
from src.data.api.exceptions import APIError, ResourceNotFoundError, ParseError

logger = logging.getLogger(__name__)

async def get_team_players(
    team_id: str,
    client: Optional[AsyncESPNClient] = None,
    incremental: bool = False
) -> List[Player]:
    """
    Get players for a specific team.
    
    Args:
        team_id: Team ID
        client: Optional AsyncESPNClient instance
        incremental: Whether to use incremental updates
        
    Returns:
        List of Player objects
        
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
            # Get players from API
            raw_players = await client.get_team_players(team_id)
            
            # Convert to Player objects
            players = []
            for raw_player in raw_players:
                try:
                    player = Player(
                        id=raw_player["id"],
                        name=raw_player["name"],
                        jersey=raw_player.get("jersey"),
                        position=raw_player.get("position", "Unknown"),
                        height=raw_player.get("height"),
                        weight=raw_player.get("weight"),
                        year=raw_player.get("year")
                    )
                    players.append(player)
                except Exception as e:
                    logger.warning(f"Failed to parse player: {e}")
            
            logger.info(f"Successfully retrieved {len(players)} players for team {team_id}")
            return players
    except ResourceNotFoundError:
        logger.error(f"Team {team_id} not found")
        raise
    except APIError as e:
        logger.error(f"Failed to get players for team {team_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting players for team {team_id}: {e}")
        raise APIError(f"Unexpected error: {str(e)}")

async def get_players_batch(
    team_ids: List[str],
    client: Optional[AsyncESPNClient] = None,
    incremental: bool = False
) -> Dict[str, List[Player]]:
    """
    Get players for multiple teams concurrently.
    
    Args:
        team_ids: List of team IDs
        client: Optional AsyncESPNClient instance
        incremental: Whether to use incremental updates
        
    Returns:
        Dictionary mapping team IDs to lists of Player objects
        
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
            # Create tasks for getting players for each team
            tasks = {team_id: get_team_players(team_id, client, incremental) for team_id in team_ids}
            
            # Execute tasks concurrently
            results = {}
            for team_id, task in tasks.items():
                try:
                    results[team_id] = await task
                except ResourceNotFoundError:
                    logger.warning(f"Team {team_id} not found")
                    results[team_id] = []
                except Exception as e:
                    logger.error(f"Failed to get players for team {team_id}: {e}")
                    results[team_id] = []
            
            # Log stats
            successful_teams = sum(1 for players in results.values() if players)
            total_players = sum(len(players) for players in results.values())
            logger.info(f"Successfully retrieved players for {successful_teams} of {len(team_ids)} teams ({total_players} total players)")
            
            return results
    except Exception as e:
        logger.error(f"Failed to get players batch: {e}")
        raise APIError(f"Batch player retrieval failed: {str(e)}")

async def get_player_stats(
    player_id: str,
    season: Optional[str] = None,
    client: Optional[AsyncESPNClient] = None
) -> Dict[str, Any]:
    """
    Get statistics for a specific player.
    
    Note: This is a placeholder for a future implementation.
    The current ESPN API client doesn't have an endpoint for player stats.
    
    Args:
        player_id: Player ID
        season: Optional season year (e.g., "2023-24")
        client: Optional AsyncESPNClient instance
        
    Returns:
        Dictionary of player statistics
        
    Raises:
        ResourceNotFoundError: If player not found
        APIError: If API request fails
    """
    # This would be implemented once the ESPN client supports this endpoint
    # For now, we'll return a placeholder
    logger.warning("Player stats endpoint not yet implemented")
    return {
        "player_id": player_id,
        "season": season or "current",
        "stats": {},
        "message": "Statistics not available - endpoint not implemented"
    } 