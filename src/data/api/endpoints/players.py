from typing import List, Dict, Any, Optional
import logging
import asyncio

from src.data.api.espn_client.client import AsyncESPNClient
from src.data.api.models.player import Player, PlayerStats
from src.data.api.exceptions import APIError, ResourceNotFoundError
from src.data.api.endpoints.common import get_async_context

logger = logging.getLogger(__name__)


def _parse_player_stats(stats_data: Dict[str, Any]) -> Optional[PlayerStats]:
    """Parse player stats from API response.

    Args:
        stats_data: Stats data from API response

    Returns:
        PlayerStats object or None if no stats data
    """
    if not stats_data:
        return None

    # Extract basic stats
    points_per_game = float(stats_data.get("ppg", 0.0))
    rebounds_per_game = float(stats_data.get("rpg", 0.0))
    assists_per_game = float(stats_data.get("apg", 0.0))

    # Create stats object
    return PlayerStats(
        points_per_game=points_per_game,
        rebounds_per_game=rebounds_per_game,
        assists_per_game=assists_per_game,
    )


def _create_player_from_response(player_data: Dict[str, Any]) -> Player:
    """Create Player object from API response.

    Args:
        player_data: Player data from API response

    Returns:
        Player object
    """
    player_id = str(player_data.get("id", ""))

    # Parse name fields
    full_name = player_data.get("displayName", "")
    first_name = player_data.get("firstName", "")
    last_name = player_data.get("lastName", "")

    # Parse team information
    team_id = None
    team_name = None
    if team := player_data.get("team", {}):
        team_id = str(team.get("id", ""))
        team_name = team.get("name", "")

    # Parse position
    position = player_data.get("position", {}).get("name", "")

    # Parse jersey number
    jersey = player_data.get("jersey", "")

    # Parse headshot URL
    headshot = player_data.get("headshot", {}).get("href", None)

    # Parse stats if available
    stats = None
    if statistics := player_data.get("statistics", []):
        if statistics and isinstance(statistics, list) and statistics:
            stats_data = statistics[0]
            stats = _parse_player_stats(stats_data)

    return Player(
        id=player_id,
        full_name=full_name,
        first_name=first_name,
        last_name=last_name,
        team_id=team_id,
        team_name=team_name,
        position=position,
        jersey=jersey,
        headshot=headshot,
        stats=stats,
    )


async def get_player_details(
    player_id: str, client: Optional[AsyncESPNClient] = None, should_close: bool = True
) -> Player:
    """Get details for a specific player.

    Args:
        player_id: Player ID
        client: Optional AsyncESPNClient instance
        should_close: Whether to close the client when done

    Returns:
        Player object

    Raises:
        ResourceNotFoundError: If the player is not found
        APIError: If there's an error with the API request
    """
    client_provided = client is not None
    client = client or AsyncESPNClient()

    try:
        async with get_async_context(client, should_close and not client_provided):
            response = await client.get_player_details(player_id)

            if not response or "athlete" not in response:
                raise ResourceNotFoundError(f"Player {player_id} not found")

            player_data = response["athlete"]
            player = _create_player_from_response(player_data)

            logger.info(f"Retrieved details for player {player_id}")
            return player
    except ResourceNotFoundError:
        logger.error(f"Player {player_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error retrieving player details for {player_id}: {e}")
        raise APIError(f"Failed to retrieve player details: {e}") from e


async def get_players_by_team(
    team_id: str, client: Optional[AsyncESPNClient] = None, should_close: bool = True
) -> List[Player]:
    """Get players for a specific team.

    Args:
        team_id: Team ID
        client: Optional AsyncESPNClient instance
        should_close: Whether to close the client when done

    Returns:
        List of Player objects

    Raises:
        ResourceNotFoundError: If the team is not found
        APIError: If there's an error with the API request
    """
    client_provided = client is not None
    client = client or AsyncESPNClient()

    try:
        async with get_async_context(client, should_close and not client_provided):
            response = await client.get_team_players(team_id)

            players = []
            for player_data in response:
                try:
                    players.append(_create_player_from_response(player_data))
                except Exception as e:
                    logger.warning(f"Failed to parse player: {e}")

            logger.info(f"Retrieved {len(players)} players for team {team_id}")
            return players
    except ResourceNotFoundError:
        logger.error(f"Team {team_id} roster not found")
        raise
    except Exception as e:
        logger.error(f"Error retrieving players for team {team_id}: {e}")
        raise APIError(f"Failed to retrieve team roster: {e}") from e


async def get_players_batch(
    player_ids: List[str], client: Optional[AsyncESPNClient] = None, should_close: bool = True
) -> List[Player]:
    """Get details for multiple players concurrently.

    Args:
        player_ids: List of player IDs
        client: Optional AsyncESPNClient instance
        should_close: Whether to close the client when done

    Returns:
        List of Player objects

    Raises:
        APIError: If there's an error with the API request
    """
    client_provided = client is not None
    client = client or AsyncESPNClient()

    try:
        async with get_async_context(client, should_close and not client_provided):
            # Create tasks for each player ID
            tasks = []
            for player_id in player_ids:
                tasks.append(get_player_details(player_id, client=client, should_close=False))

            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            players = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error retrieving player: {result}")
                else:
                    players.append(result)

            logger.info(f"Retrieved {len(players)} players in batch")
            return players
    except Exception as e:
        logger.error(f"Error retrieving players batch: {e}")
        raise APIError(f"Failed to retrieve players batch: {e}") from e


async def get_player_stats(
    player_id: str, season: Optional[str] = None, client: Optional[AsyncESPNClient] = None
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
        "message": "Statistics not available - endpoint not implemented",
    }
