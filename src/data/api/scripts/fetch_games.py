#!/usr/bin/env python3
import asyncio
import logging
import sys
import argparse
from datetime import datetime, timedelta
from typing import List, Optional

from src.data.api.espn_client import AsyncESPNClient
from src.data.api.endpoints.games import get_games_by_date_range, get_games_for_week
from src.data.api.models.game import Game

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch games from ESPN API")
    parser.add_argument("--start-date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--team-id", type=str, help="Filter by team ID")
    parser.add_argument("--week", action="store_true", help="Fetch games for the next week")
    
    return parser.parse_args()

async def fetch_games(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    team_id: Optional[str] = None,
    get_week: bool = False
):
    """Fetch and display games from the ESPN API."""
    try:
        # Create client
        client = AsyncESPNClient()
        
        # Determine date range
        if get_week:
            logger.info("Fetching games for the next week...")
            async with client:
                games = await get_games_for_week(
                    start_date=datetime.now() if not start_date else datetime.strptime(start_date, "%Y-%m-%d"),
                    team_id=team_id,
                    client=client
                )
        else:
            # Default to today if no dates provided
            if not start_date:
                today = datetime.now()
                start_date = today.strftime("%Y-%m-%d")
            
            if not end_date:
                end_date = start_date
            
            # Convert dates to format expected by API
            start_str = start_date.replace("-", "")
            end_str = end_date.replace("-", "")
            
            logger.info(f"Fetching games from {start_date} to {end_date}...")
            
            # Get games
            async with client:
                games = await get_games_by_date_range(
                    start_date=start_str,
                    end_date=end_str,
                    team_id=team_id,
                    client=client
                )
        
        # Display games
        logger.info(f"Found {len(games)} games:")
        for i, game in enumerate(games, 1):
            # Format teams and scores
            teams_str = " vs ".join([
                f"{team.name} ({team.score})" for team in game.teams
            ])
            
            # Format date
            try:
                game_date = datetime.fromisoformat(game.date.replace('Z', '+00:00'))
                date_str = game_date.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                date_str = game.date
            
            logger.info(f"{i}. [{game.status.value.upper()}] {date_str} - {game.name}")
            logger.info(f"   {teams_str}")
        
        return games
    except Exception as e:
        logger.error(f"Failed to fetch games: {e}")
        return []

async def main():
    """Main entry point."""
    args = parse_args()
    
    await fetch_games(
        start_date=args.start_date,
        end_date=args.end_date,
        team_id=args.team_id,
        get_week=args.week
    )

if __name__ == "__main__":
    asyncio.run(main()) 