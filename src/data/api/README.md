# ESPN API Client for NCAA Basketball

This module provides a robust, asynchronous client for fetching NCAA basketball data from ESPN's public API endpoints.

## Features

- **Asynchronous Processing**: Built with `asyncio` and `aiohttp` for efficient request handling
- **Adaptive Rate Limiting**: Automatically adjusts request rates to avoid API throttling
- **Retry Logic**: Uses `tenacity` for robust retry behavior on transient errors
- **Incremental Data Collection**: Tracks last update timestamps to optimize API calls
- **Rich Error Handling**: Provides detailed exception types for different error scenarios
- **Type-Safe Models**: Uses `pydantic` for data validation and serialization

## Architecture

The module follows a layered design:

```
src/data/api/
├── async_client.py        # Base HTTP client with retry logic
├── rate_limiter.py        # Rate limiting implementation
├── espn_client.py         # ESPN-specific client
├── exceptions.py          # API exception classes
├── metadata.py            # Incremental data tracking utilities
├── endpoints/             # Endpoint implementations
│   ├── teams.py           # Team data endpoints
│   ├── games.py           # Game data endpoints
│   └── players.py         # Player data endpoints
├── models/                # Data models
│   ├── team.py            # Team data models
│   ├── game.py            # Game data models
│   └── player.py          # Player data models
└── scripts/               # Utility scripts
    ├── fetch_teams.py     # Script to fetch team data
    └── fetch_games.py     # Script to fetch game data
```

## Usage Examples

### Fetching Teams

```python
import asyncio
from src.data.api.espn_client import AsyncESPNClient
from src.data.api.endpoints.teams import get_all_teams

async def fetch_teams():
    async with AsyncESPNClient() as client:
        teams = await get_all_teams(client=client)
        print(f"Retrieved {len(teams)} teams")
        return teams

# Run the example
teams = asyncio.run(fetch_teams())
```

### Fetching Games

```python
import asyncio
from datetime import datetime, timedelta
from src.data.api.espn_client import AsyncESPNClient
from src.data.api.endpoints.games import get_games_by_date_range

async def fetch_recent_games():
    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    async with AsyncESPNClient() as client:
        games = await get_games_by_date_range(
            start_date=start_str,
            end_date=end_str,
            client=client
        )
        print(f"Retrieved {len(games)} games")
        return games

# Run the example
games = asyncio.run(fetch_recent_games())
```

### Incremental Data Collection

```python
import asyncio
from src.data.api.espn_client import AsyncESPNClient
from src.data.api.endpoints.teams import get_all_teams

async def fetch_teams_incremental():
    async with AsyncESPNClient() as client:
        # Only fetch teams that have been updated since last call
        teams = await get_all_teams(client=client, incremental=True)
        if teams:
            print(f"Retrieved {len(teams)} updated teams")
        else:
            print("No team updates available")
        return teams

# Run the example
teams = asyncio.run(fetch_teams_incremental())
```

### Error Handling

```python
import asyncio
from src.data.api.espn_client import AsyncESPNClient
from src.data.api.exceptions import (
    APIError, ResourceNotFoundError, RateLimitError
)

async def fetch_team_with_error_handling(team_id):
    async with AsyncESPNClient() as client:
        try:
            team = await client.get_team(team_id)
            print(f"Team: {team.get('name')}")
            return team
        except ResourceNotFoundError:
            print(f"Team {team_id} not found")
        except RateLimitError:
            print("Rate limit exceeded, try again later")
        except APIError as e:
            print(f"API error: {e}")
        return None

# Run the example
team = asyncio.run(fetch_team_with_error_handling("invalid-id"))
```

## Command-Line Tools

The module includes command-line tools for common operations:

### Fetch Teams

```bash
# Basic usage
python -m src.data.api.scripts.fetch_teams

# With options
python -m src.data.api.scripts.fetch_teams --output data/teams.json --detailed --incremental
```

### Fetch Games

```bash
# Basic usage
python -m src.data.api.scripts.fetch_games

# With options
python -m src.data.api.scripts.fetch_games --start-date 20230101 --end-date 20230107 --team 52 --detailed
```

## Development Guidelines

When contributing to this module, please follow these guidelines:

1. **Follow TDD**: Write tests before implementing features
2. **Use asyncio patterns**: All API operations should be asynchronous
3. **Handle errors properly**: Use appropriate exception types
4. **Document your code**: Include docstrings and type hints
5. **Support incremental updates**: Consider caching and metadata tracking

## Testing

Tests are located in the `tests/data/api` directory. Run them with pytest:

```bash
python -m pytest tests/data/api
```

## Extended Documentation

For more detailed documentation, please refer to the [API Documentation](../../../docs/api-documentation.md) in the docs directory. 