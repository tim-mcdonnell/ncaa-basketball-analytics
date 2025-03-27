#!/usr/bin/env python3
import asyncio
import logging
import sys

from src.data.api.espn_client.client import AsyncESPNClient
from src.data.api.endpoints.teams import get_all_teams, get_team_details

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def fetch_teams():
    """Fetch and display teams from the ESPN API."""
    logger.info("Fetching teams from ESPN API...")

    try:
        # Create client
        client = AsyncESPNClient()

        # Get all teams
        async with client:
            teams = await get_all_teams(client)

        # Display teams
        logger.info(f"Found {len(teams)} teams:")
        for i, team in enumerate(teams[:10], 1):  # Show first 10 teams
            logger.info(f"{i}. {team.name} (ID: {team.id})")

        if len(teams) > 10:
            logger.info(f"...and {len(teams) - 10} more teams")

        return teams
    except Exception as e:
        logger.error(f"Failed to fetch teams: {e}")
        return []


async def fetch_team_details(team_id: str):
    """Fetch and display details for a specific team."""
    logger.info(f"Fetching details for team ID: {team_id}")

    try:
        # Create client
        client = AsyncESPNClient()

        # Get team details
        async with client:
            team = await get_team_details(team_id, client)

        # Display team details
        logger.info(f"Team: {team.name}")
        logger.info(f"  ID: {team.id}")
        logger.info(f"  Abbreviation: {team.abbreviation}")
        logger.info(f"  Location: {team.location}")
        logger.info(
            f"  Record: {team.record.summary} ({team.record.wins} wins, {team.record.losses} losses)"
        )

        return team
    except Exception as e:
        logger.error(f"Failed to fetch team details: {e}")
        return None


async def main():
    """Main entry point."""
    # If team ID is provided as argument, fetch details for that team
    if len(sys.argv) > 1:
        team_id = sys.argv[1]
        await fetch_team_details(team_id)
    else:
        # Otherwise, fetch all teams
        teams = await fetch_teams()

        if teams:
            # Fetch details for the first team
            first_team = teams[0]
            logger.info(f"\nFetching details for first team: {first_team.name}")
            await fetch_team_details(first_team.id)


if __name__ == "__main__":
    asyncio.run(main())
