---
title: Data Storage Implementation
description: Technical specification for data storage in Phase 01 MVP
---

# Data Storage Implementation

This document provides technical details for implementing the data storage component of Phase 01 MVP.

## 🎯 Overview

**Background:** Reliable data storage is essential for our NCAA basketball analytics pipeline, providing the foundation for all downstream analysis, feature engineering, and model training.

**Objective:** Establish a reliable and efficient database architecture using DuckDB to store NCAA basketball data.

**Scope:** This component will handle data ingestion from the API, schema management, data validation, and provide a consistent interface for other components to access the data.

## 📐 Technical Requirements

### Architecture

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

### Data Versioning and Lineage

1. Schema versioning must:
   - Track schema changes over time
   - Support backward compatibility
   - Enable schema evolution without data loss
   - Document changes between versions

2. Data lineage tracking must:
   - Record data sources for all records
   - Track transformations applied to data
   - Link processed data back to raw sources

## 🧪 Testing Requirements

### Test-Driven Development Process

1. **RED Phase**:
   - Write failing tests for database schema creation
   - Write failing tests for repository operations
   - Create tests for data validation and quality checks
   - Develop tests for ingestion pipelines

2. **GREEN Phase**:
   - Implement database schema and minimum functionality to pass tests
   - Develop repository operations to satisfy test requirements
   - Build data validation and quality check implementations
   - Create ingestion pipelines that pass defined tests

3. **REFACTOR Phase**:
   - Optimize repository operations for performance
   - Enhance data validation with more comprehensive checks
   - Improve error handling and edge case management
   - Refactor for code clarity and maintainability

### Test Cases

- [ ] Test `test_database_initialization`: Verify database file creation and schema setup
- [ ] Test `test_raw_team_ingestion`: Verify raw team data is correctly stored
- [ ] Test `test_raw_game_ingestion`: Verify raw game data is correctly stored
- [ ] Test `test_raw_player_ingestion`: Verify raw player data is correctly stored
- [ ] Test `test_dim_team_population`: Verify dimensional team data extraction
- [ ] Test `test_fact_game_population`: Verify fact game data extraction
- [ ] Test `test_dim_player_population`: Verify dimensional player data extraction
- [ ] Test `test_fact_player_stats_population`: Verify player stats extraction
- [ ] Test `test_data_validation`: Verify data validation constraints are enforced
- [ ] Test `test_referential_integrity`: Verify foreign key relationships
- [ ] Test `test_incremental_updates`: Verify incremental update mechanism
- [ ] Test `test_schema_evolution`: Verify schema migration mechanism

### Database Testing

```python
# Example test for database initialization
def test_database_initialization():
    # Arrange
    db_path = "test_ncaa.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Act
    db = Database(db_path)
    db.initialize_schema()
    
    # Assert
    assert os.path.exists(db_path)
    
    # Verify all tables exist
    tables = db.list_tables()
    required_tables = [
        "raw_teams", "raw_games", "raw_players",
        "dim_teams", "dim_players", "dim_seasons",
        "fact_games", "fact_player_stats"
    ]
    for table in required_tables:
        assert table in tables
```

### Real-World Testing

- Run: `python -m src.data.storage.scripts.init_database`
- Verify: Database file is created with all expected tables

- Run: `python -m src.data.storage.scripts.ingest_sample_data`
- Verify:
  1. Raw tables contain the expected number of records
  2. Dimension and fact tables are populated
  3. Referential integrity is maintained

## 📄 Documentation Requirements

- [ ] Document database schema in `docs/architecture/data-table-structures.md`
- [ ] Create data flow diagrams showing ingestion process in `docs/architecture/data-flow.md`
- [ ] Document repository pattern usage in `docs/guides/data-access.md`
- [ ] Document data validation rules in `docs/architecture/data-validation.md`
- [ ] Add schema migration process to `docs/guides/schema-evolution.md`

### Code Documentation Standards

- All repository classes must have:
  - Class-level docstrings explaining purpose and usage
  - Method docstrings with parameters and return values
  - Example usage in docstrings where appropriate

- Schema definitions must have:
  - Table and column comments
  - Constraint explanations
  - Version information

## 🛠️ Implementation Process

1. Set up initial project structure and test framework for database
2. Implement and test database connection and schema creation
3. Implement and test raw layer repositories
4. Implement and test dimensional model and fact table repositories
5. Develop and test data transformation from raw to processed layer
6. Implement and test data validation and quality checks
7. Add schema migration support
8. Implement data lineage tracking
9. Performance optimization and query tuning
10. Integration with API client for end-to-end data flow

## ✅ Acceptance Criteria

- [ ] All specified tests pass, including integration tests
- [ ] Database schema correctly implements all required tables
- [ ] Repository pattern provides clean data access interface
- [ ] Data ingestion successfully stores API data
- [ ] Data transformation correctly populates dimensional model
- [ ] Data validation enforces data quality rules
- [ ] Referential integrity is maintained
- [ ] Schema versioning and migration mechanism works
- [ ] Data lineage tracking captures source information
- [ ] Performance meets query response time targets
- [ ] Documentation is complete and accurate
- [ ] Code meets project quality standards (passes linting and typing)

## Database Schema Definition

```sql
-- Raw Layer Tables
CREATE TABLE IF NOT EXISTS raw_teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id VARCHAR NOT NULL,
    raw_data JSON NOT NULL,
    source_url VARCHAR NOT NULL,
    collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_games (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id VARCHAR NOT NULL,
    raw_data JSON NOT NULL,
    source_url VARCHAR NOT NULL,
    collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_players (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id VARCHAR NOT NULL,
    raw_data JSON NOT NULL,
    source_url VARCHAR NOT NULL,
    collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR NOT NULL
);

-- Dimensional Model Tables
CREATE TABLE IF NOT EXISTS dim_teams (
    team_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    short_name VARCHAR,
    conference VARCHAR,
    division VARCHAR,
    location VARCHAR,
    mascot VARCHAR,
    wins INTEGER,
    losses INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_data_id UUID NOT NULL REFERENCES raw_teams(id)
);

CREATE TABLE IF NOT EXISTS dim_players (
    player_id VARCHAR PRIMARY KEY,
    first_name VARCHAR NOT NULL,
    last_name VARCHAR NOT NULL,
    team_id VARCHAR NOT NULL REFERENCES dim_teams(team_id),
    position VARCHAR,
    jersey_number INTEGER,
    year VARCHAR,
    height INTEGER, -- in inches
    weight INTEGER, -- in pounds
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_data_id UUID NOT NULL REFERENCES raw_players(id)
);

CREATE TABLE IF NOT EXISTS fact_games (
    game_id VARCHAR PRIMARY KEY,
    home_team_id VARCHAR NOT NULL REFERENCES dim_teams(team_id),
    away_team_id VARCHAR NOT NULL REFERENCES dim_teams(team_id),
    game_date DATE NOT NULL,
    game_time TIME,
    location VARCHAR,
    status VARCHAR NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_data_id UUID NOT NULL REFERENCES raw_games(id)
);

CREATE TABLE IF NOT EXISTS fact_player_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id VARCHAR NOT NULL REFERENCES dim_players(player_id),
    game_id VARCHAR NOT NULL REFERENCES fact_games(game_id),
    minutes_played INTEGER,
    points INTEGER NOT NULL DEFAULT 0,
    rebounds INTEGER NOT NULL DEFAULT 0,
    assists INTEGER NOT NULL DEFAULT 0,
    steals INTEGER NOT NULL DEFAULT 0,
    blocks INTEGER NOT NULL DEFAULT 0,
    turnovers INTEGER NOT NULL DEFAULT 0,
    field_goals_made INTEGER NOT NULL DEFAULT 0,
    field_goals_attempted INTEGER NOT NULL DEFAULT 0,
    three_pointers_made INTEGER NOT NULL DEFAULT 0,
    three_pointers_attempted INTEGER NOT NULL DEFAULT 0,
    free_throws_made INTEGER NOT NULL DEFAULT 0,
    free_throws_attempted INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (player_id, game_id)
);
```

## Implementation Examples

### Repository Implementation Example

```python
class TeamRepository:
    """Repository for accessing team data in the dimensional model."""
    
    def __init__(self, db_connection):
        """Initialize with a database connection."""
        self.db = db_connection
    
    def get_by_id(self, team_id: str) -> Optional[Team]:
        """
        Retrieve a team by ID.
        
        Args:
            team_id: The unique identifier for the team
            
        Returns:
            Team object if found, None otherwise
        """
        query = """
        SELECT * FROM dim_teams 
        WHERE team_id = ?
        """
        result = self.db.execute(query, (team_id,)).fetchone()
        if not result:
            return None
        return Team(**result)
    
    def get_by_conference(self, conference: str) -> List[Team]:
        """
        Retrieve all teams in a specific conference.
        
        Args:
            conference: The conference name
            
        Returns:
            List of Team objects in the specified conference
        """
        query = """
        SELECT * FROM dim_teams 
        WHERE conference = ?
        ORDER BY name
        """
        results = self.db.execute(query, (conference,)).fetchall()
        return [Team(**row) for row in results]
    
    def create(self, team: Team) -> Team:
        """
        Create a new team record.
        
        Args:
            team: Team object with data to insert
            
        Returns:
            The created Team with any database-generated fields
        """
        query = """
        INSERT INTO dim_teams (
            team_id, name, short_name, conference, division,
            location, mascot, wins, losses, raw_data_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING *
        """
        params = (
            team.team_id, team.name, team.short_name, 
            team.conference, team.division, team.location,
            team.mascot, team.wins, team.losses, team.raw_data_id
        )
        result = self.db.execute(query, params).fetchone()
        return Team(**result)
```

## Architecture Alignment

This data storage implementation aligns with the specifications in the architecture documentation:

1. Uses DuckDB for data storage as specified in tech-stack.md
2. Implements the medallion architecture (raw, dimension/fact layers)
3. Uses repository pattern for data access
4. Follows the project structure for database components
5. Provides data lineage tracking from raw to processed data
6. Implements proper validation for data integrity

## Integration Points

- **Input**: Receives data from the API client component
- **Output**: Provides structured data access for feature engineering
- **Configuration**: Reads database configuration from config files
- **Versioning**: Supports schema evolution over time
- **Validation**: Enforces data quality rules

## Technical Challenges

1. **Schema Evolution**: Supporting changes to data structure over time
2. **Data Reconciliation**: Managing incremental updates consistently
3. **Performance**: Optimizing query performance for analytical workloads
4. **Data Integrity**: Ensuring quality and consistency of stored data

## Success Metrics

1. **Data Integrity**: Zero data quality issues in processed layer
2. **Performance**: Sub-second query response for common operations
3. **Scalability**: Handles growing dataset sizes efficiently
4. **Reliability**: Consistent data access with proper error handling 