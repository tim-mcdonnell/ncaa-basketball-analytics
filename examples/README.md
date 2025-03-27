# NCAA Basketball Analytics Examples

This directory contains standalone example scripts that demonstrate the key components and functionality of the NCAA Basketball Analytics project. Each example is self-contained and can be run independently without importing from the main project code.

## Available Examples

### 1. ESPN API Client (`espn_api_example.py`)

Demonstrates how to use the ESPN API client to retrieve NCAA basketball data, including:
- Asynchronous API requests with rate limiting
- Error handling and retries with exponential backoff
- Fetching teams, games, schedules, and player rosters

**Run with:** `python examples/espn_api_example.py`

### 2. Data Storage (`data_storage_example.py`)

Shows how to use the DuckDB-based data storage component, including:
- Database connection management
- Schema definition for raw and dimensional data models
- Repository pattern for data access
- Data transformation from raw to dimensional format
- SQL queries with Polars integration

**Run with:** `python examples/data_storage_example.py`

### 3. Feature Framework (`feature_framework_example.py`)

Illustrates the feature engineering framework, including:
- Feature definition and metadata management
- Feature dependencies and computation
- Storing and retrieving feature values from the database
- Creating complex features based on simpler ones

**Run with:** `python examples/feature_framework_example.py`

## Integration Between Examples

The data storage and feature framework examples use the same database file (`example.duckdb` in the examples directory), allowing you to see how different components interact:

1. Run `data_storage_example.py` first to set up the database schema and populate it with sample data
2. Then run `feature_framework_example.py` to see how features are built on top of the stored data

This demonstrates how the components work together in the full system.

## Usage Notes

1. The examples share a common database file (`example.duckdb`) for integration
2. Examples print informative output to the console to explain what's happening
3. The examples are designed to be read alongside their execution to understand the concepts

To get the most value from these examples, we recommend:
1. Running the example scripts in the suggested order
2. Reading through the code to understand the implementation patterns
3. Using these patterns when implementing new components in the main project
