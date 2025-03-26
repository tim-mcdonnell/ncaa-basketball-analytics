---
title: ESPN API Integration
description: Technical specification for ESPN API integration in Phase 1 MVP
---

# ESPN API Integration

This document provides technical details for implementing the ESPN API integration component of Phase 1 MVP.

## Overview

The ESPN API integration component will establish a reliable data collection framework from ESPN's public and authenticated APIs, focusing on college basketball data. The component will handle authentication, rate limiting, error handling, and data validation, using asynchronous processing for efficient data collection.

## Architecture

```
src/
└── data/
    └── api/
        ├── __init__.py
        ├── client.py             # Base API client with auth and rate limiting
        ├── async_client.py       # Asynchronous base client implementation
        ├── espn_client.py        # ESPN-specific client implementation
        ├── rate_limiter.py       # Adaptive rate limiting implementation
        ├── endpoints/            # Endpoint implementations
        │   ├── __init__.py
        │   ├── teams.py          # Team data endpoints
        │   ├── games.py          # Game data endpoints
        │   └── players.py        # Player data endpoints
        └── models/               # Data models for API responses
            ├── __init__.py
            ├── team.py
            ├── game.py
            └── player.py
```

## Technical Requirements

### Asynchronous API Client

1. Base API client class must implement:
   - Asynchronous request handling with aiohttp
   - Adaptive rate limiting with dynamic concurrency adjustment
   - Exponential backoff with jitter for retries
   - Request/response logging
   - Error handling with appropriate retries
   - Timeout configuration

2. Adaptive rate limiting must implement:
   - Dynamic concurrency level adjustment based on API responses
   - Automatic throttling when rate limits are detected
   - Progressive ramping up of concurrency after successful responses
   - Configurable min/max concurrency limits

3. ESPN-specific client must implement:
   - API key management (if required)
   - ESPN-specific endpoints with async methods
   - Error response handling
   - Response caching where appropriate

### Data Models

1. Pydantic models for all API responses to ensure:
   - Type validation
   - Default values for missing fields
   - Conversion of ESPN data format to internal format

### Endpoints

1. Teams endpoint:
   - Asynchronously retrieve list of all NCAA basketball teams
   - Retrieve detailed information for specific teams
   - Handle conference and division filtering

2. Games endpoint:
   - Asynchronously retrieve game schedule data
   - Retrieve game results and statistics
   - Support filtering by date range, team, or conference
   - Handle game status (scheduled, in-progress, completed)

3. Players endpoint:
   - Asynchronously retrieve player roster by team
   - Retrieve player statistics
   - Handle active/inactive player status

### Incremental Data Collection

1. Track last successful data collection timestamp
2. Support incremental updates based on:
   - New games added to schedule
   - Game status changes
   - Updated statistics

### Concurrency Management

1. Implement adaptive concurrency control:
   - Start with a conservative number of concurrent requests
   - Dynamically adjust based on response patterns
   - Detect rate limiting responses and reduce concurrency
   - Gradually increase concurrency after successful responses

2. Request management:
   - Queue requests to control execution
   - Batch related requests when possible
   - Group requests by endpoint type for better concurrency management

### Testing Requirements

1. Mock ESPN API responses for all endpoints
2. Test adaptive rate limiting behavior
3. Test error handling and retries with exponential backoff
4. Test concurrent request management
5. Test data validation for various API response scenarios
6. Test incremental data collection logic

## Usage Examples

```python
# Async client initialization
import asyncio
from src.data.api.espn_client import AsyncESPNClient

async def fetch_all_data():
    async with AsyncESPNClient() as client:
        # Fetch all teams concurrently
        teams = await client.get_teams()
        
        # Concurrently fetch multiple date ranges
        date_ranges = [
            (datetime(2023, 11, 1), datetime(2023, 11, 30)),
            (datetime(2023, 12, 1), datetime(2023, 12, 31))
        ]
        all_games = []
        games_tasks = [
            client.get_games(start_date=start, end_date=end)
            for start, end in date_ranges
        ]
        all_games_results = await asyncio.gather(*games_tasks)
        for games in all_games_results:
            all_games.extend(games)
            
        # Concurrently fetch player data for multiple teams
        team_ids = ["59", "127", "248"]  # Example team IDs
        player_tasks = [
            client.get_team_players(team_id=team_id)
            for team_id in team_ids
        ]
        all_players_results = await asyncio.gather(*player_tasks)
        
        return teams, all_games, all_players_results

# Run in an async context
results = asyncio.run(fetch_all_data())
```

## Implementation of Adaptive Rate Limiting

```python
class AdaptiveRateLimiter:
    """Dynamically adjusts concurrency levels based on success/failure patterns"""
    def __init__(self, initial=10, min_limit=1, max_limit=50):
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.current_limit = initial
        self.semaphore = asyncio.Semaphore(initial)
        self.success_streak = 0
        self.failure_streak = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        await self.semaphore.acquire()
    
    async def release(self, success=True):
        self.semaphore.release()
        async with self.lock:
            if success:
                self.success_streak += 1
                self.failure_streak = 0
                # Increase concurrency after consecutive successes
                if self.success_streak >= 10 and self.current_limit < self.max_limit:
                    self._increase_concurrency()
            else:
                self.failure_streak += 1
                self.success_streak = 0
                # Decrease concurrency after consecutive failures
                if self.failure_streak >= 3 and self.current_limit > self.min_limit:
                    self._decrease_concurrency()
```

## Implementation Approach

1. First implement and test the async base API client with adaptive rate limiting
2. Implement and test the retry mechanism using tenacity
3. Next implement the teams endpoint with concurrent fetching
4. Then implement the games endpoint with appropriate filtering
5. Finally implement the players endpoint
6. Add incremental data collection logic with proper concurrency control

## Integration with Airflow

The API client will be integrated with Airflow through a PythonOperator:

```python
def run_async_espn_collection():
    import asyncio
    from src.data.api.espn_client import AsyncESPNClient
    from src.data.storage.repositories.raw.team_repo import RawTeamRepository
    
    async def fetch_and_store():
        async with AsyncESPNClient() as client:
            teams = await client.get_teams()
            # Store in database
            # ...
    
    # Run async collection within Airflow task
    asyncio.run(fetch_and_store())

# In DAG definition
collect_espn_data = PythonOperator(
    task_id='collect_espn_data',
    python_callable=run_async_espn_collection,
    # ...
)
```

## Integration Points

- **Output**: API client will provide validated data to the data storage component
- **Configuration**: Client will read rate limiting and API parameters from config files
- **Logging**: All API activity will be logged using the project's logging system
- **Storage**: Raw API responses will be stored as JSON in DuckDB `raw_*` tables

## Technical Challenges

1. **API Stability**: ESPN APIs may change without notice; design should handle this gracefully
2. **Rate Limiting**: Need to dynamically adjust request rate to avoid being blocked
3. **Data Consistency**: NCAA data can be inconsistent; validation must be robust
4. **Concurrency Management**: Effectively managing concurrent requests without overwhelming the API

## Success Metrics

1. **Reliability**: >99% success rate for API requests with appropriate retries
2. **Efficiency**: Minimal collection time through optimal concurrency
3. **Coverage**: All required data points successfully collected
4. **Adaptability**: System automatically adjusts to API rate limits and availability 