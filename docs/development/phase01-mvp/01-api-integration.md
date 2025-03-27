---
title: ESPN API Integration
description: Technical specification for ESPN API integration in Phase 01 MVP
---

# ESPN API Integration

This document provides technical details for implementing the ESPN API integration component of Phase 01 MVP.

## 🎯 Overview

**Background:** The project requires reliable NCAA basketball data collection to power all downstream analytics and predictions.

**Objective:** Establish a reliable data collection framework from ESPN's public and authenticated APIs, focusing on college basketball data.

**Scope:** This component will handle authentication, rate limiting, error handling, data validation, and use asynchronous processing for efficient data collection.

## 📐 Technical Requirements

### Architecture

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

## 🧪 Testing Requirements

### Test-Driven Development Process

1. **RED Phase**:
   - Write failing tests for each client component before implementation
   - Create mock ESPN API responses for testing
   - Establish endpoint test coverage requirements

2. **GREEN Phase**:
   - Implement minimum code to pass tests
   - Ensure each component passes its unit tests
   - Integration tests verify components work together

3. **REFACTOR Phase**:
   - Optimize code while keeping tests passing
   - Improve error handling and edge cases
   - Enhance performance where possible

### Test Cases

- [ ] Test `test_base_client_initialization`: Verify client initializes with correct configuration
- [ ] Test `test_rate_limiter_throttling`: Verify rate limiter reduces concurrency when limits detected
- [ ] Test `test_exponential_backoff`: Verify retry mechanism applies correct backoff
- [ ] Test `test_team_endpoint_retrieval`: Verify teams endpoint returns expected data
- [ ] Test `test_games_endpoint_filtering`: Verify games endpoint filters correctly
- [ ] Test `test_players_endpoint_by_team`: Verify player roster retrieval by team
- [ ] Test `test_error_handling`: Verify client handles API errors gracefully
- [ ] Test `test_incremental_updates`: Verify client correctly handles incremental updates

### Mock API Testing

Create mock ESPN API responses in `tests/data/api/responses/` for:
- Team listings with complete hierarchy
- Game schedules with different statuses
- Player statistics with various edge cases
- Error responses and rate limiting scenarios

### Real-World Testing

- Run: `python -m src.data.api.scripts.fetch_teams`
- Verify: Client successfully retrieves team data without errors

- Run: `python -m src.data.api.scripts.fetch_games --start-date 2023-11-01 --end-date 2023-11-30`
- Verify:
  1. All games in the date range are retrieved
  2. Game data includes expected fields
  3. No rate limiting errors occur

## 📄 Documentation Requirements

- [ ] Update API client usage examples in `docs/guides/api-usage.md`
- [ ] Document ESPN API endpoints and limitations in `docs/architecture/data-sources.md`
- [ ] Add detailed code documentation for all public methods
- [ ] Create sequence diagrams showing client operation in `docs/architecture/api-flow.md`

### Code Documentation Standards

- All public methods must have docstrings with:
  - Description of functionality
  - Parameter descriptions and types
  - Return value description and type
  - Exception information

## 🛠️ Implementation Process

1. Set up project structure and test framework for API client
2. Implement and test base async client with rate limiting (without ESPN specifics)
3. Implement and test retry logic with exponential backoff
4. Implement and test ESPN-specific client implementation
5. Implement and test teams endpoint with concurrent fetching
6. Implement and test games endpoint with filtering
7. Implement and test players endpoint
8. Add incremental data collection logic
9. Integrate with logging framework
10. Performance optimization and concurrent request tuning

## ✅ Acceptance Criteria

- [ ] All specified tests pass, including integration tests
- [ ] Client can retrieve all required data types from ESPN API
- [ ] Rate limiting effectively prevents API request failures
- [ ] Concurrent requests optimize data collection speed
- [ ] Error handling gracefully manages API failures
- [ ] Response data is correctly validated and transformed
- [ ] Incremental updates work correctly
- [ ] Documentation is complete and accurate
- [ ] Code meets project quality standards (passes linting and typing)
- [ ] Real-world testing demonstrates reliable operation

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

## Integration with Airflow

The API client will be integrated with Airflow DAGs for scheduled data collection:

1. Separate DAG for different data collection tasks
2. Task-specific error handling and notifications
3. Incremental data collection scheduling
4. Metrics collection for monitoring

## Architecture Alignment

This API integration implementation aligns with the specifications in the architecture documentation:

1. Uses aiohttp for asynchronous API requests as specified in tech-stack.md
2. Implements adaptive rate limiting and concurrency control as outlined
3. Uses tenacity for retry mechanisms with exponential backoff
4. Follows the project structure for API client organization
5. Supports the data flow into the raw layer of the medallion architecture
6. Implements proper validation for API responses

## Implementation Timeline

The API integration component should be implemented early in the Phase 1 development cycle, as it provides the data foundation for all downstream components.

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
