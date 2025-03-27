"""Team endpoints for ESPN API."""

from typing import List, Optional, Dict, Any
import logging
import asyncio

from src.data.api.espn_client.client import AsyncESPNClient
from src.data.api.models.team import Team, TeamRecord
from src.data.api.exceptions import APIError, ResourceNotFoundError
from src.data.api.endpoints.common import get_async_context

logger = logging.getLogger(__name__)


def _parse_team_record(raw_record: Optional[Dict[str, Any]]) -> Optional[TeamRecord]:
    """
    Parse team record from API response.

    Args:
        raw_record: Raw record data from API

    Returns:
        TeamRecord object or None if not available
    """
    if not raw_record:
        return TeamRecord()  # Return default record

    # Handle different record formats in ESPN API
    if isinstance(raw_record, str):
        # Parse string like "10-5"
        try:
            wins, losses = raw_record.split("-")
            return TeamRecord(summary=raw_record, wins=int(wins), losses=int(losses))
        except (ValueError, IndexError):
            return TeamRecord(summary=raw_record)

    # Handle nested record format
    if isinstance(raw_record, dict):
        if "items" in raw_record and raw_record["items"]:
            # Extract summary (e.g., "10-2")
            summary = raw_record["items"][0].get("summary", "0-0")

            # Try to parse wins and losses
            try:
                wins, losses = summary.split("-")
                return TeamRecord(summary=summary, wins=int(wins), losses=int(losses))
            except (ValueError, IndexError):
                return TeamRecord(summary=summary)

    # Default to empty record if can't parse
    return TeamRecord()


async def get_all_teams(
    client: Optional[AsyncESPNClient] = None,
) -> List[Team]:
    """
    Get all NCAA basketball teams.

    Args:
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
        async with get_async_context(client, should_close):
            # Get teams from API
            raw_teams = await client.get_teams()

            # Parse teams into Team objects
            teams = []
            for raw_team in raw_teams:
                try:
                    # Extract required fields
                    team_id = raw_team["id"]
                    name = raw_team.get("displayName", raw_team.get("name", ""))
                    abbreviation = raw_team.get("abbreviation", "")

                    # Parse team record
                    record = _parse_team_record(raw_team.get("record"))

                    # Create team object
                    team = Team(
                        id=team_id,
                        name=name,
                        abbreviation=abbreviation,
                        record=record,
                        location=raw_team.get("location"),
                        logo=raw_team.get("logo"),
                    )
                    teams.append(team)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse team data: {e}")
                    continue

            logger.info(f"Successfully retrieved {len(teams)} teams")
            return teams
    except APIError as e:
        logger.error(f"Failed to get teams: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting teams: {e}")
        raise APIError(f"Failed to get teams: {str(e)}")


async def get_team_details(
    team_id: str,
    client: Optional[AsyncESPNClient] = None,
) -> Team:
    """
    Get detailed information for a specific team.

    Args:
        team_id: Team ID
        client: Optional AsyncESPNClient instance

    Returns:
        Team object

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
        async with get_async_context(client, should_close):
            # Get team details from API
            raw_team = await client.get_team_details(team_id)

            # Get team data from the 'team' field if present
            if "team" in raw_team:
                raw_team = raw_team["team"]

            # Extract required fields
            name = raw_team.get("displayName", raw_team.get("name", ""))
            abbreviation = raw_team.get("abbreviation", "")

            # Parse team record
            record = _parse_team_record(raw_team.get("record"))

            # Create team object
            team = Team(
                id=team_id,
                name=name,
                abbreviation=abbreviation,
                record=record,
                location=raw_team.get("location"),
                logo=raw_team.get(
                    "logo",
                    raw_team.get("logos", [{}])[0].get("href") if raw_team.get("logos") else None,
                ),
            )

            logger.info(f"Successfully retrieved details for team {team_id}: {name}")
            return team
    except ResourceNotFoundError:
        logger.error(f"Team {team_id} not found")
        raise
    except APIError as e:
        logger.error(f"Failed to get team {team_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting team {team_id}: {e}")
        raise APIError(f"Failed to get team {team_id}: {str(e)}")


async def get_teams_batch(
    team_ids: List[str],
    client: Optional[AsyncESPNClient] = None,
) -> List[Team]:
    """
    Get details for multiple teams in parallel.

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
        async with get_async_context(client, should_close):
            # Create tasks for each team ID
            tasks = []
            for team_id in team_ids:
                task = asyncio.create_task(get_team_details(team_id, client))
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and log errors
            teams = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    if isinstance(result, ResourceNotFoundError):
                        logger.warning(f"Team {team_ids[i]} not found")
                    else:
                        logger.error(f"Failed to get team {team_ids[i]}: {result}")
                else:
                    teams.append(result)

            logger.info(f"Successfully retrieved {len(teams)} of {len(team_ids)} teams in batch")
            return teams
    except Exception as e:
        logger.error(f"Unexpected error in batch team retrieval: {e}")
        raise APIError(f"Failed to get teams in batch: {str(e)}")
