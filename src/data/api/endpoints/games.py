"""Game endpoints for ESPN API."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncio

from src.data.api.espn_client.client import AsyncESPNClient
from src.data.api.models.game import Game, GameStatus, TeamScore
from src.data.api.exceptions import APIError, ResourceNotFoundError, ParseError
from src.data.api.endpoints.common import get_async_context

logger = logging.getLogger(__name__)


def _parse_date_string(date_str: str) -> datetime:
    """Parse date string into datetime object.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        datetime object
    """
    return datetime.strptime(date_str, "%Y-%m-%d")


def _parse_game_status(game_data: Dict[str, Any]) -> GameStatus:
    """Parse game status from API response.

    Args:
        game_data: Game data from API response

    Returns:
        GameStatus object
    """
    status_data = {}

    # Check for competitions -> status -> type -> completed
    competitions = game_data.get("competitions", [])
    if competitions and len(competitions) > 0:
        comp_status = competitions[0].get("status", {})
        status_type = comp_status.get("type", {})

        status_data["is_completed"] = status_type.get("completed", False)
        status_data["status_text"] = status_type.get("description", "")

    # Check top-level status as fallback
    elif "status" in game_data and "type" in game_data["status"]:
        status_type = game_data["status"].get("type", {})
        status_data["is_completed"] = status_type.get("completed", False)
        status_data["status_text"] = status_type.get("description", "")

    return GameStatus(**status_data)


def _parse_team_score(team_data: Dict[str, Any], is_home: bool = False) -> TeamScore:
    """Parse team score from API response.

    Args:
        team_data: Team data from API response
        is_home: Whether this is the home team

    Returns:
        TeamScore object
    """
    if not team_data:
        raise ValueError("Team data is required")

    # Get team ID from team object or directly from competitor
    if "team" in team_data and isinstance(team_data["team"], dict):
        team_id = str(team_data["team"].get("id", ""))
        team_name = team_data["team"].get("name", "")
    else:
        team_id = str(team_data.get("id", ""))
        team_name = team_data.get("name", "")

    score = team_data.get("score", 0)

    # Handle case where score might be a string
    if isinstance(score, str) and score.isdigit():
        score = int(score)
    elif not isinstance(score, int):
        score = 0

    return TeamScore(team_id=team_id, team_name=team_name, score=score, is_home=is_home)


def _create_game_from_response(game_data: Dict[str, Any]) -> Game:
    """Create Game object from API response.

    Args:
        game_data: Game data from API response

    Returns:
        Game object

    Raises:
        ParseError: If required data is missing
    """
    if not game_data:
        raise ParseError("Game data is required")

    game_id = str(game_data.get("id", ""))
    if not game_id:
        raise ParseError("Game ID is required")

    # Parse date
    date_str = game_data.get("date", "")
    try:
        date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        date_obj = None

    name = game_data.get("name", "")

    # Parse status
    status = _parse_game_status(game_data)

    # Parse teams
    competitions = game_data.get("competitions", [])
    if not competitions:
        raise ParseError(f"No competitions found for game {game_id}")

    competition = competitions[0]
    competitors = competition.get("competitors", [])
    if len(competitors) < 2:
        raise ParseError(f"Not enough competitors found for game {game_id}")

    # Find home and away teams
    home_team = None
    away_team = None

    for competitor in competitors:
        is_home = competitor.get("homeAway", "").lower() == "home"
        team_score = _parse_team_score(competitor, is_home)

        if is_home:
            home_team = team_score
        else:
            away_team = team_score

    # If home/away not specified, use first as home, second as away
    if home_team is None and away_team is None and len(competitors) >= 2:
        home_team = _parse_team_score(competitors[0], True)
        away_team = _parse_team_score(competitors[1], False)

    # Parse venue and attendance
    venue = competition.get("venue", {}).get("fullName", None)
    attendance = competition.get("attendance", None)

    # Create Game object
    return Game(
        id=game_id,
        date=date_obj,
        name=name,
        status=status,
        home_team=home_team,
        away_team=away_team,
        venue=venue,
        attendance=attendance,
    )


async def get_games_by_date(
    date_str: str, client: Optional[AsyncESPNClient] = None, should_close: bool = True
) -> List[Game]:
    """Get games for a specific date.

    Args:
        date_str: Date string in YYYY-MM-DD format
        client: Optional AsyncESPNClient instance
        should_close: Whether to close the client when done

    Returns:
        List of Game objects

    Raises:
        APIError: If there's an error with the API request
    """
    client_provided = client is not None
    client = client or AsyncESPNClient()

    try:
        async with get_async_context(client, should_close and not client_provided):
            date_formatted = date_str.replace("-", "")  # Convert to YYYYMMDD format for API
            events = await client.get_games(date_str=date_formatted)

            games = []
            for game_data in events:
                try:
                    games.append(_create_game_from_response(game_data))
                except Exception as e:
                    logger.warning(f"Failed to parse game: {e}")

            logger.info(f"Retrieved {len(games)} games for date {date_str}")
            return games
    except Exception as e:
        logger.error(f"Error retrieving games by date {date_str}: {e}")
        raise APIError(f"Failed to retrieve games by date: {e}") from e


async def get_games_by_team(
    team_id: str,
    limit: int = 100,
    client: Optional[AsyncESPNClient] = None,
    should_close: bool = True,
) -> List[Game]:
    """Get games for a specific team.

    Args:
        team_id: Team ID
        limit: Maximum number of games to retrieve
        client: Optional AsyncESPNClient instance
        should_close: Whether to close the client when done

    Returns:
        List of Game objects

    Raises:
        APIError: If there's an error with the API request
    """
    client_provided = client is not None
    client = client or AsyncESPNClient()

    try:
        async with get_async_context(client, should_close and not client_provided):
            events = await client.get_games(team_id=team_id, limit=limit)

            games = []
            for game_data in events:
                try:
                    games.append(_create_game_from_response(game_data))
                except Exception as e:
                    logger.warning(f"Failed to parse game: {e}")

            logger.info(f"Retrieved {len(games)} games for team {team_id}")
            return games
    except Exception as e:
        logger.error(f"Error retrieving games for team {team_id}: {e}")
        raise APIError(f"Failed to retrieve games for team: {e}") from e


async def get_game_details(
    game_id: str, client: Optional[AsyncESPNClient] = None, should_close: bool = True
) -> Game:
    """Get details for a specific game.

    Args:
        game_id: Game ID
        client: Optional AsyncESPNClient instance
        should_close: Whether to close the client when done

    Returns:
        Game object

    Raises:
        ResourceNotFoundError: If the game is not found
        APIError: If there's an error with the API request
    """
    client_provided = client is not None
    client = client or AsyncESPNClient()

    try:
        async with get_async_context(client, should_close and not client_provided):
            response = await client.get_game(game_id)

            if not response:
                raise ResourceNotFoundError(f"Game {game_id} not found")

            game = _create_game_from_response(response)

            logger.info(f"Retrieved details for game {game_id}")
            return game
    except ResourceNotFoundError:
        logger.error(f"Game {game_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error retrieving game details for {game_id}: {e}")
        raise APIError(f"Failed to retrieve game details: {e}") from e


async def get_games_by_date_range(
    start_date: str,
    end_date: str,
    team_id: Optional[str] = None,
    client: Optional[AsyncESPNClient] = None,
    should_close: bool = True,
) -> List[Game]:
    """Get games within a date range, optionally filtered by team.

    Args:
        start_date: Start date string in YYYY-MM-DD format
        end_date: End date string in YYYY-MM-DD format
        team_id: Optional team ID to filter games
        client: Optional AsyncESPNClient instance
        should_close: Whether to close the client when done

    Returns:
        List of Game objects

    Raises:
        APIError: If there's an error with the API request
    """
    client_provided = client is not None
    client = client or AsyncESPNClient()

    try:
        async with get_async_context(client, should_close and not client_provided):
            start = _parse_date_string(start_date)
            end = _parse_date_string(end_date)

            # Create a list of dates in the range
            date_range = []
            current = start
            while current <= end:
                date_range.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

            # If team_id is provided, get games by team instead
            if team_id:
                return await get_games_by_team(
                    team_id, limit=100, client=client, should_close=False
                )

            # Otherwise get games for each date in the range
            tasks = []
            for date_str in date_range:
                tasks.append(get_games_by_date(date_str, client=client, should_close=False))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and flatten the list
            games = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error retrieving games for a date: {result}")
                else:
                    games.extend(result)

            logger.info(f"Retrieved {len(games)} games for date range {start_date} to {end_date}")
            return games
    except Exception as e:
        logger.error(f"Error retrieving games for date range {start_date} to {end_date}: {e}")
        raise APIError(f"Failed to retrieve games by date range: {e}") from e


async def get_games_for_week(
    start_date: Optional[datetime] = None,
    team_id: Optional[str] = None,
    client: Optional[AsyncESPNClient] = None,
    incremental: bool = False,
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
        incremental=incremental,
    )


async def get_recent_games(
    team_id: str,
    days: int = 30,
    client: Optional[AsyncESPNClient] = None,
    incremental: bool = False,
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
        incremental=incremental,
    )
