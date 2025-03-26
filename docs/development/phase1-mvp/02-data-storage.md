---
title: Data Storage Implementation
description: Technical specification for data storage in Phase 1 MVP
---

# Data Storage Implementation

This document provides technical details for implementing the data storage component of Phase 1 MVP.

## Overview

The data storage component will establish a reliable and efficient database architecture using DuckDB to store NCAA basketball data. It will handle data ingestion from the API, schema management, data validation, and provide a consistent interface for other components to access the data.

## Architecture

```
src/
└── data/
    └── storage/
        ├── __init__.py
        ├── db.py              # Database connection and management
        ├── schema.py          # Database schema definitions
        ├── migrations/        # Schema migrations
        │   └── __init__.py
        ├── repositories/      # Data access repositories
        │   ├── __init__.py
        │   ├── raw/
        │   │   ├── __init__.py
        │   │   ├── team_repo.py
        │   │   ├── game_repo.py
        │   │   └── player_repo.py
        │   ├── dim/
        │   │   ├── __init__.py
        │   │   ├── team_repo.py
        │   │   ├── player_repo.py
        │   │   └── season_repo.py
        │   └── fact/
        │       ├── __init__.py
        │       ├── game_repo.py
        │       └── player_stats_repo.py
        └── models/            # Internal data models
            ├── __init__.py
            ├── raw/
            │   ├── __init__.py
            │   ├── team.py
            │   ├── game.py
            │   └── player.py
            ├── dim/
            │   ├── __init__.py
            │   ├── team.py
            │   ├── player.py
            │   └── season.py
            └── fact/
                ├── __init__.py
                ├── game.py
                └── player_stats.py
```

## Technical Requirements

### Database Schema

#### Raw Layer Tables

1. Raw Teams table (`raw_teams`):
   - id: UUID (primary key, generated)
   - team_id: VARCHAR (from API)
   - raw_data: JSON (complete API response)
   - source_url: VARCHAR (API endpoint)
   - collected_at: TIMESTAMP (data collection time)
   - processing_version: VARCHAR (API client version)

2. Raw Games table (`raw_games`):
   - id: UUID (primary key, generated)
   - game_id: VARCHAR (from API)
   - raw_data: JSON (complete API response)
   - source_url: VARCHAR (API endpoint)
   - collected_at: TIMESTAMP (data collection time)
   - processing_version: VARCHAR (API client version)

3. Raw Players table (`raw_players`):
   - id: UUID (primary key, generated)
   - player_id: VARCHAR (from API)
   - raw_data: JSON (complete API response)
   - source_url: VARCHAR (API endpoint)
   - collected_at: TIMESTAMP (data collection time)
   - processing_version: VARCHAR (API client version)

#### Processed Layer Tables (Dimensional Model)

1. Teams dimension table (`dim_teams`):
   - team_id: VARCHAR (primary key)
   - name: VARCHAR
   - short_name: VARCHAR
   - conference: VARCHAR
   - division: VARCHAR
   - location: VARCHAR
   - mascot: VARCHAR
   - wins: INTEGER
   - losses: INTEGER
   - created_at: TIMESTAMP
   - updated_at: TIMESTAMP
   - raw_data_id: UUID (foreign key to raw_teams)

2. Games fact table (`fact_games`):
   - game_id: VARCHAR (primary key)
   - home_team_id: VARCHAR (foreign key)
   - away_team_id: VARCHAR (foreign key)
   - game_date: DATE
   - game_time: TIME
   - location: VARCHAR
   - status: VARCHAR
   - home_score: INTEGER
   - away_score: INTEGER
   - created_at: TIMESTAMP
   - updated_at: TIMESTAMP
   - raw_data_id: UUID (foreign key to raw_games)

3. Players dimension table (`dim_players`):
   - player_id: VARCHAR (primary key)
   - first_name: VARCHAR
   - last_name: VARCHAR
   - team_id: VARCHAR (foreign key)
   - position: VARCHAR
   - jersey_number: INTEGER
   - year: VARCHAR
   - height: INTEGER
   - weight: INTEGER
   - created_at: TIMESTAMP
   - updated_at: TIMESTAMP
   - raw_data_id: UUID (foreign key to raw_players)

4. Player Stats fact table (`fact_player_stats`):
   - id: UUID (primary key, generated)
   - player_id: VARCHAR (foreign key)
   - game_id: VARCHAR (foreign key)
   - minutes_played: INTEGER
   - points: INTEGER
   - rebounds: INTEGER
   - assists: INTEGER
   - steals: INTEGER
   - blocks: INTEGER
   - turnovers: INTEGER
   - field_goals_made: INTEGER
   - field_goals_attempted: INTEGER
   - three_pointers_made: INTEGER
   - three_pointers_attempted: INTEGER
   - free_throws_made: INTEGER
   - free_throws_attempted: INTEGER
   - created_at: TIMESTAMP
   - updated_at: TIMESTAMP

### Data Ingestion

1. Implement ingestion pipelines for:
   - Storing raw API responses as JSON in `raw_*` tables
   - Extracting data from raw tables to populate dimensional model
   - Incremental updates with reconciliation logic

2. Implement data transformation using Polars:
   - Parse JSON to extract structured data 
   - Handle data type conversions
   - Apply business logic and derivations

### Data Access

1. Repository pattern implementation for each layer:
   - Raw layer repositories (`raw/team_repo.py`, etc.)
   - Dimension table repositories (`dim/team_repo.py`, etc.)
   - Fact table repositories (`fact/game_repo.py`, etc.)

2. Query optimization for common access patterns:
   - Team performance metrics
   - Game results
   - Player statistics

### Data Validation and Quality

1. Implement data validation at ingestion:
   - Type validation
   - Referential integrity
   - Business rule validation

2. Implement data quality checks:
   - Completeness checks
   - Consistency checks
   - Anomaly detection

### Testing Requirements

1. Test database schema creation and migrations
2. Test data ingestion pipelines
3. Test data access repositories
4. Test data validation and quality checks
5. Test performance for large datasets

## Usage Examples

```python
# Database connection
from src.data.storage.db import get_connection

# Storing raw team data
from src.data.storage.repositories.raw.team_repo import RawTeamRepository
import json
import uuid

conn = get_connection()
raw_team_repo = RawTeamRepository(conn)

# Store raw API response
team_api_response = {...}  # JSON from API
raw_team = {
    "id": str(uuid.uuid4()),
    "team_id": "59",
    "raw_data": json.dumps(team_api_response),
    "source_url": "https://api.espn.com/v1/sports/basketball/mens-college-basketball/teams/59",
    "collected_at": datetime.now(),
    "processing_version": "1.0.0"
}
raw_team_repo.insert(raw_team)

# Extract and store processed team data
from src.data.storage.repositories.dim.team_repo import DimTeamRepository
from src.data.storage.models.dim.team import Team

team_repo = DimTeamRepository(conn)

# Create normalized team record from raw data
team = Team(
    team_id="59",
    name="Michigan Wolverines",
    short_name="MICH",
    conference="Big Ten",
    division="D1",
    location="Ann Arbor, MI",
    mascot="Wolverines",
    wins=15,
    losses=5,
    raw_data_id=raw_team["id"]
)

team_repo.insert(team)

# Querying games for a team
from src.data.storage.repositories.fact.game_repo import FactGameRepository

game_repo = FactGameRepository(conn)
michigan_games = game_repo.find_by_team_id("59")
```

## Implementation Approach

1. First implement and test database schema and connection management
2. Next implement raw data repositories for storing API responses
3. Then implement dimension table repositories for normalized data
4. Add fact table repositories with appropriate relationships
5. Implement data validation and quality checks
6. Add performance optimizations for common queries

## Integration Points

- **Input**: Storage component will receive data from API component
- **Output**: Will provide data to feature engineering component
- **Configuration**: Will read database configuration from config files
- **Logging**: All database operations will be logged using the project's logging system

## Technical Challenges

1. **Schema Evolution**: Design must accommodate changes in data structure over time
2. **Performance**: Need to optimize for both write (ingestion) and read (analysis) operations
3. **Data Integrity**: Must maintain referential integrity across multiple data sources
4. **JSON Storage**: Efficient storage and querying of JSON data in DuckDB

## Performance Considerations

1. Use appropriate indexes for common query patterns
2. Consider partitioning for large tables (games, player_stats)
3. Use batch processing for data ingestion
4. Create views for frequently accessed query patterns
5. Implement caching for repetitive queries

## Success Metrics

1. **Reliability**: Zero data loss during ingestion
2. **Performance**: Query response times <500ms for common queries
3. **Integrity**: 100% referential integrity maintained
4. **Efficiency**: Storage overhead <20% of raw data size 