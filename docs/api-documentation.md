# ESPN API Integration Documentation

This document provides information on using the ESPN API integration for NCAA Basketball data.

## Overview

The ESPN API integration module provides a robust, asynchronous client for fetching NCAA basketball data from ESPN's public API endpoints. The module offers:

- Asynchronous request handling using Python's `asyncio` and `aiohttp`
- Automatic rate limiting with adaptive strategies
- Retry logic using Tenacity for reliable requests
- Comprehensive error handling
- Incremental data collection to minimize API calls
- Data models built with Pydantic

## Installation

This module is part of the NCAA Basketball Analytics project. No additional installation is required beyond the project dependencies.

### Requirements

- Python 3.9+
- aiohttp
- tenacity
- pydantic

## Architecture

The ESPN API integration is built with a layered architecture:

1. **Core Client Layer** (`async_client.py`): Provides base HTTP functionality with retries
2. **Rate Limiting Layer** (`rate_limiter.py`): Manages API request rates to avoid throttling
3. **ESPN Client Layer** (`espn_client.py`): ESPN-specific implementation with metadata tracking
4. **Endpoint Layer** (`endpoints/`): Specific endpoint implementations
5. **Model Layer** (`models/`): Data models for different entities

## Basic Usage

### Fetching Teams

```python
import asyncio
from src.data.api.espn_client import AsyncESPNClient
from src.data.api.endpoints.teams import get_all_teams

async def fetch_teams_example():
    async with AsyncESPNClient() as client:
        # Get all NCAA basketball teams
        teams = await get_all_teams(client=client)
        print(f"Retrieved {len(teams)} teams")

        # Print team names
        for team in teams:
            print(f"{team.name} ({team.abbreviation})")

# Run the example
asyncio.run(fetch_teams_example())
```

### Fetching Games

```python
import asyncio
from datetime import datetime, timedelta
from src.data.api.espn_client import AsyncESPNClient
from src.data.api.endpoints.games import get_games_by_date_range

async def fetch_games_example():
    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    async with AsyncESPNClient() as client:
        # Get games for the last week
        games = await get_games_by_date_range(
            start_date=start_str,
            end_date=end_str,
            client=client
        )
        print(f"Retrieved {len(games)} games from {start_str} to {end_str}")

# Run the example
asyncio.run(fetch_games_example())
```

### Fetching Players

```python
import asyncio
from src.data.api.espn_client import AsyncESPNClient
from src.data.api.endpoints.teams import get_all_teams
from src.data.api.endpoints.players import get_team_players

async def fetch_players_example():
    async with AsyncESPNClient() as client:
        # Get teams first
        teams = await get_all_teams(client=client)
        if not teams:
            print("No teams found")
            return

        # Get players for the first team
        first_team = teams[0]
        team_id = first_team.get("id")
        team_name = first_team.get("name", "Unknown")

        print(f"Fetching players for {team_name} (ID: {team_id})")
        players = await get_team_players(team_id, client=client)

        print(f"Retrieved {len(players)} players for {team_name}")
        # Print player names and positions
        for player in players:
            print(f"{player.name} - {player.position} (#{player.jersey})")

# Run the example
asyncio.run(fetch_players_example())
```

## Command-Line Utilities

The module includes command-line utilities for fetching data:

### Fetch Teams

```bash
# Basic usage
python -m src.data.api.scripts.fetch_teams

# With options
python -m src.data.api.scripts.fetch_teams --output data/teams.json --detailed --incremental
```

### Fetch Games

```bash
# Basic usage (fetches last 7 days)
python -m src.data.api.scripts.fetch_games

# With date range and team filter
python -m src.data.api.scripts.fetch_games --start-date 20230301 --end-date 20230310 --team 52 --detailed
```

## Error Handling

The API integration provides several exception types for handling different error scenarios:

- `APIError`: Base exception for all API-related errors
- `RateLimitError`: Raised when rate limits are exceeded
- `AuthenticationError`: Raised for authentication failures
- `ResourceNotFoundError`: Raised when requested resources don't exist
- `ValidationError`: Raised for invalid input parameters
- `ServiceUnavailableError`: Raised when the API service is unavailable
- `ParseError`: Raised when response parsing fails
- `TimeoutError`: Raised when requests time out

Example error handling:

```python
from src.data.api.exceptions import APIError, ResourceNotFoundError, RateLimitError

try:
    # API call here
    result = await client.get_team("invalid-id")
except ResourceNotFoundError:
    print("Team not found")
except RateLimitError:
    print("Rate limit exceeded, try again later")
except APIError as e:
    print(f"API error occurred: {e}")
```

## Incremental Data Collection

The API client supports incremental data collection to minimize API calls:

```python
# Fetch only updates since last retrieval
teams = await get_all_teams(client=client, incremental=True)

# Same for games and players
games = await get_games_by_date_range(
    start_date="20230301",
    end_date="20230310",
    incremental=True,
    client=client
)
```

Incremental collection tracks timestamps in a metadata file at `data/metadata/espn_metadata.json`.

## Rate Limiting

The client includes adaptive rate limiting to prevent API throttling:

```python
# Customize rate limits
client = AsyncESPNClient(
    rate_limit_calls=5,  # Maximum calls in period
    rate_limit_period=60.0  # Period in seconds
)
```

## Advanced Configuration

For advanced scenarios, the client can be customized:

```python
from src.data.api.espn_client import AsyncESPNClient

# Customize the client
client = AsyncESPNClient(
    # Base URL (if different from default)
    base_url="https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball",

    # Retry configuration
    max_retries=5,
    retry_min_wait=2.0,
    retry_max_wait=30.0,
    retry_factor=2.0,

    # Timeout configuration
    timeout=60.0,

    # Rate limiting configuration
    rate_limit_calls=10,
    rate_limit_period=60.0,

    # Metadata configuration
    metadata_dir="custom/metadata/path",
    metadata_file="custom_metadata.json"
)
```

## Troubleshooting

Common issues and solutions:

1. **Rate Limiting Errors**: If you receive rate limiting errors, try:
   - Reducing the number of concurrent requests
   - Increasing the rate limit period
   - Adding delays between batches of requests

2. **Timeout Errors**: For timeout errors:
   - Increase the timeout value
   - Ensure good network connectivity
   - Consider breaking requests into smaller batches

3. **Parse Errors**: If you encounter parse errors:
   - Check the API documentation for changes
   - Verify the response structure matches expectations
   - Consider using more flexible parsing approaches

## Future Plans

Planned enhancements for the ESPN API integration include:

- Support for NCAA tournament brackets and schedules
- Enhanced statistics collection
- Historical data archiving
- Custom data transformations for analysis

## ESPNApiClient Adapter

The `ESPNApiClient` adapter provides a compatibility layer between the existing `ESPNClient` implementation and the interface expected by Airflow operators and prediction components.

### Usage

```python
from src.data.api.espn_client import ESPNApiClient

# Initialize the client
client = ESPNApiClient()

# Get teams data
teams = client.get_teams(season="2022-23")

# Get games data
games = client.get_games(
    start_date="2023-01-01",
    end_date="2023-01-31",
    team_id="MICH",
    limit=100
)

# Get players data for a specific team
players = client.get_players(team_id="MICH")

# Get player statistics
player_stats = client.get_player_stats(player_id="4430185")
```

### API Reference

#### ESPNApiClient

```python
class ESPNApiClient:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ESPN API client adapter.

        Args:
            config_path: Optional path to configuration file
        """

    def get_teams(self, season: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get teams data from ESPN API.

        Args:
            season: Basketball season (e.g., '2022-23')

        Returns:
            List of team dictionaries
        """

    def get_games(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get games data from ESPN API.

        Args:
            start_date: Start date for games (YYYY-MM-DD)
            end_date: End date for games (YYYY-MM-DD)
            team_id: Team ID to filter by
            limit: Maximum number of games to return

        Returns:
            List of game dictionaries
        """

    def get_players(self, team_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get players data from ESPN API.

        Args:
            team_id: Optional team ID to filter by

        Returns:
            List of player dictionaries
        """

    def get_player_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        player_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get player statistics from ESPN API.

        Args:
            start_date: Start date for stats (YYYY-MM-DD)
            end_date: End date for stats (YYYY-MM-DD)
            player_id: Optional player ID to filter by

        Returns:
            List of player statistics dictionaries
        """
```
