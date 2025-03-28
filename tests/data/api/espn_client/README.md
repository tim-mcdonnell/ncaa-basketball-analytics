# ESPN API Client Tests

This directory contains tests for the ESPN API client components, including the synchronous client, asynchronous client, and adapter classes.

## Testing Approach

The testing in this directory follows these principles:

1. **Unit Tests**: All components are tested in isolation with mock dependencies.
2. **Interface Testing**: Tests verify that the public interfaces behave as expected.
3. **Error Handling**: Tests verify appropriate error handling and edge cases.
4. **Configuration**: Tests verify that configuration options are correctly applied.

## Key Test Files

- `test_adapter.py`: Tests for the ESPNApiClient adapter, which provides a compatibility layer for Airflow operators.
- `test_client.py`: Tests for the core ESPNClient and AsyncESPNClient classes.
- `test_teams.py`: Tests for the teams endpoint functionality.
- `test_games.py`: Tests for the games endpoint functionality.
- `test_players.py`: Tests for the players endpoint functionality.

## Running Tests

Run all ESPN client tests:

```bash
python -m pytest tests/data/api/espn_client/
```

Run a specific test file:

```bash
python -m pytest tests/data/api/espn_client/test_adapter.py
```

Run with verbose output:

```bash
python -m pytest tests/data/api/espn_client/test_adapter.py -v
```

## ESPNApiClient Adapter

The `ESPNApiClient` adapter is specifically designed to provide a compatibility layer between the existing ESPN client implementation and the interface expected by Airflow operators and prediction components. The tests for this adapter verify that:

1. It correctly delegates calls to the underlying `ESPNClient`.
2. It transforms data as necessary to match the expected format.
3. It correctly handles all options and parameters.
4. It provides proper error handling.

The adapter simplifies usage in the context of the Airflow operators and ensures consistent interfaces throughout the codebase.
