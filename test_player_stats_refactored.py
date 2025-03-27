#!/usr/bin/env python3
"""Test script for refactored player stats functionality."""

import asyncio
import logging

from src.data.api.espn_client.client import AsyncESPNClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_player_stats():
    """Test the player stats functionality."""
    try:
        # Create client and use it in a context manager
        async with AsyncESPNClient() as client:
            # Use a known player ID (this example uses a player ID that should be stable)
            player_id = "4278137"  # A sample player ID

            logger.info(f"Getting stats for player with ID: {player_id}...")

            # Test the get_player_stats method directly
            stats = await client.get_player_stats(player_id)

            logger.info(f"Player stats response: {stats}")

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_player_stats())
