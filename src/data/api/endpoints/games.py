from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime, timedelta

from src.data.api.espn_client import AsyncESPNClient
from src.data.api.models.game import Game, GameList, GameStatus, TeamScore
from src.data.api.exceptions import APIError, ResourceNotFoundError, ParseError

logger = logging.getLogger(__name__)

async def get_games_by_date_range(
    start_date: str,
    end_date: str,
    team_id: Optional[str] = None,
    client: Optional[AsyncESPNClient] = None,
    incremental: bool = False
) -> List[Game]:
    """
    Get games within a date range.
    
    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        team_id: Optional team ID to filter games
        client: Optional AsyncESPNClient instance
        incremental: Whether to use incremental updates
        
    Returns:
        List of Game objects
        
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
            # Get games from API with incremental flag for optimization
            raw_games = await client.get_games(
                start_date=start_date,
                end_date=end_date,
                team_id=team_id,
                incremental=incremental
            )
            
            # Convert to Game objects
            games = []
            for raw_game in raw_games:
                try:
                    # Create team scores
                    team_scores = []
                    for team_data in raw_game.get("teams", []):
                        team_scores.append(TeamScore(
                            id=team_data["id"],
                            name=team_data["name"],
                            score=team_data.get("score", "0")
                        ))
                    
                    # Map status
                    status = GameStatus.SCHEDULED
                    raw_status = raw_game.get("status", "scheduled")
                    if raw_status == "completed":
                        status = GameStatus.COMPLETED
                    elif raw_status == "in_progress":
                        status = GameStatus.IN_PROGRESS
                    elif raw_status == "postponed":
                        status = GameStatus.POSTPONED
                    elif raw_status == "canceled":
                        status = GameStatus.CANCELED
                    
                    # Create Game object
                    game = Game(
                        id=raw_game["id"],
                        date=raw_game["date"],
                        name=raw_game["name"],
                        short_name=raw_game.get("short_name"),
                        status=status,
                        teams=team_scores
                    )
                    games.append(game)
                except Exception as e:
                    logger.warning(f"Failed to parse game: {e}")
            
            logger.info(f"Successfully retrieved {len(games)} games for date range {start_date}-{end_date}")
            return games
    except APIError as e:
        logger.error(f"Failed to get games for date range {start_date}-{end_date}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting games: {e}")
        raise APIError(f"Unexpected error: {str(e)}")

async def get_games_for_week(
    start_date: Optional[datetime] = None,
    team_id: Optional[str] = None,
    client: Optional[AsyncESPNClient] = None,
    incremental: bool = False
) -> List[Game]:
    """
    Get games for a one-week period.
    
    Args:
        start_date: Optional start date (defaults to today)
        team_id: Optional team ID to filter games
        client: Optional AsyncESPNClient instance
        incremental: Whether to use incremental updates
        
    Returns:
        List of Game objects
        
    Raises:
        APIError: If API request fails
    """
    # Default to today if start date not provided
    if start_date is None:
        start_date = datetime.now()
    
    # Calculate end date (one week from start)
    end_date = start_date + timedelta(days=7)
    
    # Format dates
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    return await get_games_by_date_range(
        start_date=start_str,
        end_date=end_str,
        team_id=team_id,
        client=client,
        incremental=incremental
    )

async def get_recent_games(
    team_id: str,
    days: int = 30,
    client: Optional[AsyncESPNClient] = None,
    incremental: bool = False
) -> List[Game]:
    """
    Get recent games for a team.
    
    Args:
        team_id: Team ID
        days: Number of days in the past to include
        client: Optional AsyncESPNClient instance
        incremental: Whether to use incremental updates
        
    Returns:
        List of Game objects
        
    Raises:
        APIError: If API request fails
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    return await get_games_by_date_range(
        start_date=start_str,
        end_date=end_str,
        team_id=team_id,
        client=client,
        incremental=incremental
    )

async def get_game_details(
    game_id: str,
    client: Optional[AsyncESPNClient] = None
) -> Optional[Game]:
    """
    Get detailed information for a specific game.
    
    Args:
        game_id: Game ID
        client: Optional AsyncESPNClient instance
        
    Returns:
        Game object or None if not found
        
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
            # This would require a specific endpoint for game details
            # For now, we'll get today's games and find the one we want
            # In a real implementation, we would have a dedicated endpoint
            
            # Get today's date
            today = datetime.now()
            yesterday = today - timedelta(days=1)
            tomorrow = today + timedelta(days=1)
            
            # Format dates
            start_str = yesterday.strftime("%Y%m%d")
            end_str = tomorrow.strftime("%Y%m%d")
            
            # Get games
            games = await get_games_by_date_range(
                start_date=start_str,
                end_date=end_str,
                client=client
            )
            
            # Find the game by ID
            for game in games:
                if game.id == game_id:
                    logger.info(f"Found game {game_id}: {game.name}")
                    return game
            
            # Game not found in recent games
            logger.warning(f"Game {game_id} not found in recent games")
            return None
    except APIError as e:
        logger.error(f"Failed to get game details for {game_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting game details for {game_id}: {e}")
        raise APIError(f"Unexpected error: {str(e)}") 