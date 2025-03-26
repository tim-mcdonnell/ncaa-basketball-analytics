#!/usr/bin/env python3
"""
Fetch NCAA basketball games using the ESPN API client.

This script demonstrates how to use the ESPN API client to fetch games
for a specified date range, with options for incremental updates
and proper error handling.
"""

import asyncio
import argparse
import logging
import sys
import json
from typing import List, Dict, Any, Optional
import os
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.data.api.espn_client import AsyncESPNClient
from src.data.api.exceptions import APIError
from src.data.api.endpoints.games import get_games_by_date_range, get_game_details

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fetch_games.log')
    ]
)
logger = logging.getLogger(__name__)

def get_date_range(days_back: int = 7) -> tuple:
    """
    Get start and end dates for the date range.
    
    Args:
        days_back: Number of days to look back
        
    Returns:
        Tuple of (start_date, end_date) in YYYYMMDD format
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates as YYYYMMDD
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    return start_str, end_str

async def fetch_games(
    output_file: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    team_id: Optional[str] = None,
    incremental: bool = False,
    limit: Optional[int] = None,
    detailed: bool = False
) -> None:
    """
    Fetch games from the ESPN API and save to a JSON file.
    
    Args:
        output_file: Path to save the output JSON file
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        team_id: Optional team ID to filter games
        incremental: Whether to use incremental updates
        limit: Optional limit on number of games to fetch
        detailed: Whether to fetch detailed info for each game
    """
    # If dates not provided, use the last week
    if not start_date or not end_date:
        start_date, end_date = get_date_range()
        logger.info(f"Using default date range: {start_date} to {end_date}")
    
    try:
        async with AsyncESPNClient() as client:
            logger.info(
                f"Fetching games from {start_date} to {end_date} "
                f"(team={team_id}, incremental={incremental}, "
                f"limit={limit}, detailed={detailed})..."
            )
            
            # Get games for the date range
            all_games = await get_games_by_date_range(
                start_date=start_date,
                end_date=end_date,
                team_id=team_id,
                client=client,
                incremental=incremental
            )
            
            if not all_games:
                logger.info("No games to fetch or no updates needed.")
                return
            
            logger.info(f"Retrieved {len(all_games)} games")
            
            # Apply limit if specified
            if limit and limit > 0:
                all_games = all_games[:limit]
                logger.info(f"Limited to {limit} games")
            
            # If detailed info requested, fetch details for each game
            if detailed:
                games_with_details = []
                for i, game in enumerate(all_games):
                    game_id = game.get("id")
                    if not game_id:
                        logger.warning(f"Game at index {i} has no ID, skipping details")
                        games_with_details.append(game)
                        continue
                    
                    try:
                        logger.info(f"Fetching details for game {game_id} ({i+1}/{len(all_games)})")
                        game_details = await get_game_details(game_id, client=client)
                        games_with_details.append(game_details)
                    except Exception as e:
                        logger.error(f"Error fetching details for game {game_id}: {e}")
                        # Include the basic game info anyway
                        games_with_details.append(game)
                
                all_games = games_with_details
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save games to file
            with open(output_file, 'w') as f:
                json.dump(all_games, f, indent=2)
            
            logger.info(f"Saved {len(all_games)} games to {output_file}")
            
    except APIError as e:
        logger.error(f"API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description="Fetch NCAA basketball games from ESPN API")
    parser.add_argument("--output", "-o", default="data/games.json", help="Output JSON file path")
    parser.add_argument("--start-date", "-s", help="Start date (YYYYMMDD format)")
    parser.add_argument("--end-date", "-e", help="End date (YYYYMMDD format)")
    parser.add_argument("--team", "-t", help="Filter by team ID")
    parser.add_argument("--incremental", "-i", action="store_true", help="Use incremental updates")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of games to fetch")
    parser.add_argument("--detailed", "-d", action="store_true", help="Fetch detailed game information")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        asyncio.run(fetch_games(
            output_file=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            team_id=args.team,
            incremental=args.incremental,
            limit=args.limit,
            detailed=args.detailed
        ))
    except Exception as e:
        logger.error(f"Failed to fetch games: {e}")
        sys.exit(1)
        
    logger.info("Fetch completed successfully")

if __name__ == "__main__":
    main() 