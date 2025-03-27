"""
NCAA Basketball Analytics - ESPN API Example

This script demonstrates the basic usage of the ESPN API client for retrieving NCAA basketball data.
It shows how to fetch teams, games, and players with asynchronous processing and rate limiting.

NOTE: This is a standalone example that doesn't require importing from the project.
"""

import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== Rate Limiting Implementation ====================


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, rate_limit: int = 10, time_period: float = 60.0):
        """
        Initialize rate limiter.

        Args:
            rate_limit: Maximum number of requests per time period
            time_period: Time period in seconds
        """
        self.rate_limit = rate_limit
        self.time_period = time_period
        self.request_timestamps = []

    async def acquire(self) -> None:
        """
        Acquire permission to make a request, blocking if necessary to respect rate limits.
        """
        now = datetime.now().timestamp()

        # Remove timestamps older than the time period
        self.request_timestamps = [
            ts for ts in self.request_timestamps if now - ts < self.time_period
        ]

        # If at rate limit, wait until oldest request expires
        if len(self.request_timestamps) >= self.rate_limit:
            oldest = min(self.request_timestamps)
            wait_time = max(0, self.time_period - (now - oldest))
            logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)

            # Recursive call to check again after waiting
            return await self.acquire()

        # Add current timestamp and proceed
        self.request_timestamps.append(now)


# ==================== ESPN API Client Implementation ====================


@dataclass
class ESPNConfig:
    """Configuration for ESPN API client."""

    base_url: str = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
    )
    rate_limit: int = 10
    time_period: float = 60.0
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class AsyncESPNClient:
    """Asynchronous client for ESPN API."""

    def __init__(self, config: Optional[ESPNConfig] = None):
        """
        Initialize ESPN API client.

        Args:
            config: Configuration for the client
        """
        self.config = config or ESPNConfig()
        self.rate_limiter = RateLimiter(
            rate_limit=self.config.rate_limit, time_period=self.config.time_period
        )
        self.session = None

    async def __aenter__(self):
        """Create session when entering context."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session when exiting context."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the ESPN API.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        url = f"{self.config.base_url}/{endpoint}"
        params = params or {}

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Acquire permission from rate limiter
                await self.rate_limiter.acquire()

                # Make the request
                logger.debug(f"Making request to {url} with params {params}")
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Request attempt {attempt} failed: {str(e)}")

                if attempt < self.config.max_retries:
                    # Calculate backoff with jitter
                    delay = self.config.retry_delay * (2 ** (attempt - 1))
                    jitter = delay * 0.1 * (asyncio.get_event_loop().time() % 1.0)
                    backoff = delay + jitter

                    logger.info(f"Retrying in {backoff:.2f} seconds")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"Request to {url} failed after {self.config.max_retries} attempts"
                    )
                    raise

    async def get_team(self, team_id: str) -> Dict[str, Any]:
        """
        Get information for a specific team.

        Args:
            team_id: Team ID (e.g., '52' for Michigan)

        Returns:
            Team data
        """
        return await self._request(f"teams/{team_id}")

    async def get_teams(
        self, limit: int = 500, groups: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of teams.

        Args:
            limit: Maximum number of teams to retrieve
            groups: Filter by conference groups (e.g., '50' for Division I)

        Returns:
            List of teams
        """
        params = {"limit": str(limit)}
        if groups:
            params["groups"] = groups

        response = await self._request("teams", params)
        return response.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

    async def get_team_schedule(
        self, team_id: str, season: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get schedule for a specific team.

        Args:
            team_id: Team ID
            season: Season year (e.g., 2023 for 2022-2023 season)

        Returns:
            List of games
        """
        params = {}
        if season:
            params["season"] = str(season)

        response = await self._request(f"teams/{team_id}/schedule", params)
        return response.get("events", [])

    async def get_game(self, game_id: str) -> Dict[str, Any]:
        """
        Get information for a specific game.

        Args:
            game_id: Game ID

        Returns:
            Game data
        """
        return await self._request(f"scoreboard/events/{game_id}")

    async def get_team_roster(self, team_id: str) -> List[Dict[str, Any]]:
        """
        Get roster for a specific team.

        Args:
            team_id: Team ID

        Returns:
            List of players
        """
        response = await self._request(f"teams/{team_id}/roster")
        return response.get("athletes", [])


# ==================== Example usage ====================


async def main():
    """Demonstrate ESPN API client usage."""
    print("NCAA Basketball Analytics - ESPN API Example\n")

    async with AsyncESPNClient() as client:
        # Example 1: Get information for Michigan (team_id = 130)
        print("Example 1: Get Team Information")
        print("-" * 50)
        team_data = await client.get_team("130")
        team = team_data.get("team", {})
        print(f"Team: {team.get('displayName')} ({team.get('nickname')})")
        print(f"Location: {team.get('location')}")
        print(f"Conference: {team.get('conference', {}).get('name')}")
        print(f"Colors: {', '.join(team.get('color', '').split(','))}")
        print()

        # Example 2: Get Big Ten teams directly by ID
        print("Example 2: Big Ten Teams")
        print("-" * 50)

        # Known Big Ten team IDs - use fewer teams to avoid hanging
        big_ten_ids = [
            "130",  # Michigan
            "127",  # Michigan State
            "77",  # Indiana
            # Removed other teams to prevent potential hanging
        ]

        # Fetch data for selected Big Ten teams
        big_ten_teams = []
        for team_id in big_ten_ids:
            try:
                print(f"Fetching team {team_id}...")
                team_data = await client.get_team(team_id)
                if "team" in team_data:
                    big_ten_teams.append(team_data["team"])
                # Add a small delay between requests
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error fetching team {team_id}: {str(e)}")

        print(f"Found {len(big_ten_teams)} Big Ten teams:")
        for i, team in enumerate(big_ten_teams, 1):
            print(f"{i}. {team.get('displayName', 'Unknown')} ({team.get('id', 'Unknown')})")
        print()

        # Example 3: Get team schedule
        print("Example 3: Get Team Schedule")
        print("-" * 50)
        schedule = await client.get_team_schedule("130", 2023)  # Michigan's 2022-2023 schedule

        print(f"Found {len(schedule)} games in the schedule for Michigan (2022-2023):")
        for i, game in enumerate(schedule[:5], 1):  # Show just first 5
            game_date = game.get("date", "Unknown")
            competitors = game.get("competitions", [{}])[0].get("competitors", [])
            home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})

            home_name = home_team.get("team", {}).get("displayName", "Unknown")
            away_name = away_team.get("team", {}).get("displayName", "Unknown")

            print(f"{i}. {game_date[:10]}: {away_name} at {home_name}")
        print(f"... and {len(schedule) - 5} more games\n" if len(schedule) > 5 else "")

        # Example 4: Get team roster
        print("Example 4: Get Team Roster")
        print("-" * 50)
        roster = await client.get_team_roster("130")  # Michigan's roster

        print(f"Found {len(roster)} players on Michigan's roster:")
        for i, player in enumerate(roster[:5], 1):  # Show just first 5
            position = player.get("position", {}).get("name", "Unknown")
            jersey = player.get("jersey", "?")
            height = player.get("displayHeight", "?")
            weight = player.get("displayWeight", "?")

            print(f"{i}. #{jersey} {player['fullName']} - {position}, {height}, {weight}")
        print(f"... and {len(roster) - 5} more players\n" if len(roster) > 5 else "")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
