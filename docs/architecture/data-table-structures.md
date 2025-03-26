# Data Table Structures

## Overview

This document outlines the data storage and table structure approach for the NCAA Basketball Analytics project. The system uses a hybrid approach combining elements of:

1. **Medallion Architecture** (Bronze, Silver, Gold layers)
2. **Normalized Relational Model** (for dimension and fact tables)
3. **Feature Store Pattern** (for ML feature management)

## Storage Architecture

![Data Architecture](https://i.imgur.com/placeholder.png)

### Data Layers

#### Bronze Layer (Raw Data)
- JSON data stored directly from ESPN API
- Minimal transformation (except for file format conversion)
- Location: `data/raw/`

#### Silver Layer (Processed Data)
- Cleaned, validated, and structured data
- Normalized tables with relationships
- Location: `data/processed/`

#### Gold Layer (Feature Data)
- Derived features for machine learning
- Denormalized for analysis and model training
- Location: `data/features/`

## Physical Storage Strategy

The project uses a combination of:

1. **Parquet Files**: Column-oriented, compressed storage for efficient analytics
2. **DuckDB**: SQL interface for querying and transforming the Parquet files
3. **In-memory Polars DataFrames**: For active computation

## Database Schema

### Bronze Layer Tables

Raw data is stored as JSON or Parquet files with minimal schema enforcement:

```
data/raw/
├── teams/
│   ├── teams_2023.json
│   └── teams_2024.json
├── games/
│   ├── games_2023.json
│   └── games_2024.json
├── players/
│   ├── players_2023.json
│   └── players_2024.json
├── play_by_play/
│   ├── play_by_play_401474516.json
│   └── ...
├── rankings/
│   ├── rankings_2023.json
│   └── rankings_2024.json
└── statistics/
    ├── team_stats_2023.json
    └── team_stats_2024.json
```

### Silver Layer Schema

The silver layer consists of normalized tables with proper relationships:

#### Dimension Tables

##### `dim_teams`
```sql
CREATE TABLE dim_teams (
    team_id VARCHAR PRIMARY KEY,           -- Unique identifier from ESPN
    team_name VARCHAR NOT NULL,            -- Full team name
    short_name VARCHAR,                    -- Abbreviated name
    location VARCHAR,                      -- Geographic location
    mascot VARCHAR,                        -- Team mascot
    conference_id VARCHAR,                 -- Conference ID (FK to dim_conferences)
    logo_url VARCHAR,                      -- URL to team logo
    primary_color VARCHAR,                 -- Primary team color hex
    secondary_color VARCHAR,               -- Secondary team color hex
    venue_id VARCHAR,                      -- Home venue ID
    first_season INTEGER,                  -- First season in dataset
    last_season INTEGER,                   -- Last active season in dataset
    is_active BOOLEAN DEFAULT TRUE,        -- Whether team is currently active
    updated_at TIMESTAMP                   -- Last update timestamp
);
```

##### `dim_players`
```sql
CREATE TABLE dim_players (
    player_id VARCHAR PRIMARY KEY,         -- Unique identifier from ESPN
    first_name VARCHAR NOT NULL,           -- First name
    last_name VARCHAR NOT NULL,            -- Last name
    jersey_number VARCHAR,                 -- Jersey number (as string for cases like "00")
    position VARCHAR,                      -- Position code (G, F, C, etc.)
    height_inches INTEGER,                 -- Height in inches
    weight_pounds INTEGER,                 -- Weight in pounds
    birth_date DATE,                       -- Date of birth
    class_year VARCHAR,                    -- Class (Freshman, Sophomore, etc.)
    hometown_city VARCHAR,                 -- Hometown city
    hometown_state VARCHAR,                -- Hometown state/province
    headshot_url VARCHAR,                  -- URL to player headshot image
    is_active BOOLEAN DEFAULT TRUE,        -- Whether player is currently active
    updated_at TIMESTAMP                   -- Last update timestamp
);
```

##### `dim_venues`
```sql
CREATE TABLE dim_venues (
    venue_id VARCHAR PRIMARY KEY,          -- Unique identifier from ESPN
    venue_name VARCHAR NOT NULL,           -- Venue name
    city VARCHAR,                          -- City location
    state VARCHAR,                         -- State/province location
    capacity INTEGER,                      -- Seating capacity
    opened_year INTEGER,                   -- Year opened
    is_indoor BOOLEAN DEFAULT TRUE,        -- Whether venue is indoors
    latitude DOUBLE,                       -- Geographic latitude
    longitude DOUBLE,                      -- Geographic longitude
    timezone VARCHAR,                      -- Timezone identifier
    updated_at TIMESTAMP                   -- Last update timestamp
);
```

##### `dim_seasons`
```sql
CREATE TABLE dim_seasons (
    season_id INTEGER PRIMARY KEY,         -- Season identifier (year)
    display_name VARCHAR,                  -- Display name (e.g., "2023-24")
    start_date DATE,                       -- Season start date
    end_date DATE,                         -- Season end date
    is_current BOOLEAN DEFAULT FALSE       -- Whether this is current season
);
```

##### `dim_conferences`
```sql
CREATE TABLE dim_conferences (
    conference_id VARCHAR PRIMARY KEY,     -- Unique identifier from ESPN
    conference_name VARCHAR NOT NULL,      -- Full conference name
    short_name VARCHAR,                    -- Short name or abbreviation
    division VARCHAR,                      -- Division (e.g., "Division I")
    logo_url VARCHAR,                      -- URL to conference logo
    first_season INTEGER,                  -- First season in dataset
    last_season INTEGER,                   -- Last season in dataset
    is_active BOOLEAN DEFAULT TRUE,        -- Whether conference is currently active
    updated_at TIMESTAMP                   -- Last update timestamp
);
```

##### `dim_date`
```sql
CREATE TABLE dim_date (
    date_id DATE PRIMARY KEY,              -- Date in ISO format
    day_of_week INTEGER,                   -- Day of week (0=Monday, 6=Sunday)
    day_name VARCHAR,                      -- Day name (Monday, Tuesday, etc.)
    day_of_month INTEGER,                  -- Day of month (1-31)
    day_of_year INTEGER,                   -- Day of year (1-366)
    week_of_year INTEGER,                  -- Week of year
    month_num INTEGER,                     -- Month number (1-12)
    month_name VARCHAR,                    -- Month name (January, February, etc.)
    quarter INTEGER,                       -- Quarter (1-4)
    year INTEGER,                          -- Year
    is_weekend BOOLEAN,                    -- Whether date is weekend
    is_holiday BOOLEAN,                    -- Whether date is holiday
    season_id INTEGER                      -- Basketball season ID (FK to dim_seasons)
);
```

#### Fact Tables

##### `fact_games`
```sql
CREATE TABLE fact_games (
    game_id VARCHAR PRIMARY KEY,           -- Unique identifier from ESPN
    season_id INTEGER,                     -- Season ID (FK to dim_seasons)
    game_date DATE,                        -- Game date
    home_team_id VARCHAR,                  -- Home team ID (FK to dim_teams)
    away_team_id VARCHAR,                  -- Away team ID (FK to dim_teams)
    venue_id VARCHAR,                      -- Venue ID (FK to dim_venues)
    home_score INTEGER,                    -- Final home team score
    away_score INTEGER,                    -- Final away team score
    status VARCHAR,                        -- Game status (scheduled, completed, etc.)
    attendance INTEGER,                    -- Attendance count
    neutral_site BOOLEAN,                  -- Whether game is at neutral site
    conference_game BOOLEAN,               -- Whether game is conference game
    tournament_game BOOLEAN,               -- Whether game is tournament game
    tournament_round VARCHAR,              -- Tournament round if applicable
    broadcast_network VARCHAR,             -- TV network if broadcast
    game_url VARCHAR,                      -- URL to game page
    updated_at TIMESTAMP,                  -- Last update timestamp
    
    FOREIGN KEY (season_id) REFERENCES dim_seasons(season_id),
    FOREIGN KEY (game_date) REFERENCES dim_date(date_id),
    FOREIGN KEY (home_team_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (venue_id) REFERENCES dim_venues(venue_id)
);
```

##### `fact_team_game_stats`
```sql
CREATE TABLE fact_team_game_stats (
    game_id VARCHAR,                       -- Game ID (FK to fact_games)
    team_id VARCHAR,                       -- Team ID (FK to dim_teams)
    is_home BOOLEAN,                       -- Whether team is home team
    
    -- Standard stats
    points INTEGER,                        -- Total points
    field_goals_made INTEGER,              -- Field goals made
    field_goals_attempted INTEGER,         -- Field goals attempted
    field_goal_pct DOUBLE,                 -- Field goal percentage
    three_points_made INTEGER,             -- Three-pointers made
    three_points_attempted INTEGER,        -- Three-pointers attempted
    three_point_pct DOUBLE,                -- Three-point percentage
    free_throws_made INTEGER,              -- Free throws made
    free_throws_attempted INTEGER,         -- Free throws attempted
    free_throw_pct DOUBLE,                 -- Free throw percentage
    rebounds_offensive INTEGER,            -- Offensive rebounds
    rebounds_defensive INTEGER,            -- Defensive rebounds
    rebounds_total INTEGER,                -- Total rebounds
    assists INTEGER,                        -- Assists
    steals INTEGER,                        -- Steals
    blocks INTEGER,                        -- Blocks
    turnovers INTEGER,                     -- Turnovers
    fouls INTEGER,                         -- Personal fouls
    largest_lead INTEGER,                  -- Largest lead during game
    
    -- Advanced stats
    possessions INTEGER,                   -- Estimated possessions
    offensive_rating DOUBLE,               -- Points per 100 possessions
    defensive_rating DOUBLE,               -- Opponent points per 100 possessions
    effective_fg_pct DOUBLE,               -- Effective field goal percentage
    true_shooting_pct DOUBLE,              -- True shooting percentage
    pace DOUBLE,                           -- Pace (possessions per 40 minutes)
    
    updated_at TIMESTAMP,                  -- Last update timestamp
    
    PRIMARY KEY (game_id, team_id),
    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (team_id) REFERENCES dim_teams(team_id)
);
```

##### `fact_player_game_stats`
```sql
CREATE TABLE fact_player_game_stats (
    game_id VARCHAR,                       -- Game ID (FK to fact_games)
    player_id VARCHAR,                     -- Player ID (FK to dim_players)
    team_id VARCHAR,                       -- Team ID (FK to dim_teams)
    
    -- Playing time
    minutes_played INTEGER,                -- Minutes played
    is_starter BOOLEAN,                    -- Whether player started
    
    -- Standard stats
    points INTEGER,                        -- Total points
    field_goals_made INTEGER,              -- Field goals made
    field_goals_attempted INTEGER,         -- Field goals attempted
    field_goal_pct DOUBLE,                 -- Field goal percentage
    three_points_made INTEGER,             -- Three-pointers made
    three_points_attempted INTEGER,        -- Three-pointers attempted
    three_point_pct DOUBLE,                -- Three-point percentage
    free_throws_made INTEGER,              -- Free throws made
    free_throws_attempted INTEGER,         -- Free throws attempted
    free_throw_pct DOUBLE,                 -- Free throw percentage
    rebounds_offensive INTEGER,            -- Offensive rebounds
    rebounds_defensive INTEGER,            -- Defensive rebounds
    rebounds_total INTEGER,                -- Total rebounds
    assists INTEGER,                       -- Assists
    steals INTEGER,                        -- Steals
    blocks INTEGER,                        -- Blocks
    turnovers INTEGER,                     -- Turnovers
    fouls INTEGER,                         -- Personal fouls
    plus_minus INTEGER,                    -- Plus/minus rating
    
    updated_at TIMESTAMP,                  -- Last update timestamp
    
    PRIMARY KEY (game_id, player_id),
    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (player_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (team_id) REFERENCES dim_teams(team_id)
);
```

##### `fact_team_season_stats`
```sql
CREATE TABLE fact_team_season_stats (
    team_id VARCHAR,                       -- Team ID (FK to dim_teams)
    season_id INTEGER,                     -- Season ID (FK to dim_seasons)
    
    -- Record
    wins INTEGER,                          -- Total wins
    losses INTEGER,                        -- Total losses
    conf_wins INTEGER,                     -- Conference wins
    conf_losses INTEGER,                   -- Conference losses
    home_wins INTEGER,                     -- Home wins
    home_losses INTEGER,                   -- Home losses
    away_wins INTEGER,                     -- Away wins
    away_losses INTEGER,                   -- Away losses
    
    -- Aggregated stats
    games_played INTEGER,                  -- Games played
    points_per_game DOUBLE,                -- Points per game
    points_against_per_game DOUBLE,        -- Points against per game
    field_goal_pct DOUBLE,                 -- Field goal percentage
    three_point_pct DOUBLE,                -- Three-point percentage
    free_throw_pct DOUBLE,                 -- Free throw percentage
    rebounds_per_game DOUBLE,              -- Rebounds per game
    assists_per_game DOUBLE,               -- Assists per game
    steals_per_game DOUBLE,                -- Steals per game
    blocks_per_game DOUBLE,                -- Blocks per game
    turnovers_per_game DOUBLE,             -- Turnovers per game
    
    -- Advanced stats
    offensive_efficiency DOUBLE,           -- Offensive efficiency
    defensive_efficiency DOUBLE,           -- Defensive efficiency
    tempo DOUBLE,                          -- Tempo (possessions per 40 minutes)
    strength_of_schedule DOUBLE,           -- Strength of schedule rating
    
    updated_at TIMESTAMP,                  -- Last update timestamp
    
    PRIMARY KEY (team_id, season_id),
    FOREIGN KEY (team_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (season_id) REFERENCES dim_seasons(season_id)
);
```

##### `fact_player_season_stats`
```sql
CREATE TABLE fact_player_season_stats (
    player_id VARCHAR,                     -- Player ID (FK to dim_players)
    team_id VARCHAR,                       -- Team ID (FK to dim_teams)
    season_id INTEGER,                     -- Season ID (FK to dim_seasons)
    
    -- Playing time
    games_played INTEGER,                  -- Games played
    games_started INTEGER,                 -- Games started
    minutes_per_game DOUBLE,               -- Minutes per game
    
    -- Per game averages
    points_per_game DOUBLE,                -- Points per game
    rebounds_per_game DOUBLE,              -- Rebounds per game
    assists_per_game DOUBLE,               -- Assists per game
    steals_per_game DOUBLE,                -- Steals per game
    blocks_per_game DOUBLE,                -- Blocks per game
    turnovers_per_game DOUBLE,             -- Turnovers per game
    fouls_per_game DOUBLE,                 -- Fouls per game
    
    -- Shooting percentages
    field_goal_pct DOUBLE,                 -- Field goal percentage
    three_point_pct DOUBLE,                -- Three-point percentage
    free_throw_pct DOUBLE,                 -- Free throw percentage
    
    -- Advanced stats
    player_efficiency_rating DOUBLE,       -- Player Efficiency Rating (PER)
    true_shooting_pct DOUBLE,              -- True shooting percentage
    usage_rate DOUBLE,                     -- Usage rate
    
    updated_at TIMESTAMP,                  -- Last update timestamp
    
    PRIMARY KEY (player_id, team_id, season_id),
    FOREIGN KEY (player_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (team_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (season_id) REFERENCES dim_seasons(season_id)
);
```

##### `fact_play_by_play`
```sql
CREATE TABLE fact_play_by_play (
    play_id VARCHAR PRIMARY KEY,           -- Unique play identifier
    game_id VARCHAR,                       -- Game ID (FK to fact_games)
    sequence_number INTEGER,               -- Sequence number in game
    period INTEGER,                        -- Period (1, 2, OT, etc.)
    clock_minutes INTEGER,                 -- Minutes on game clock
    clock_seconds INTEGER,                 -- Seconds on game clock
    play_type VARCHAR,                     -- Type of play
    play_text VARCHAR,                     -- Text description
    team_id VARCHAR,                       -- Team ID (FK to dim_teams)
    player_id VARCHAR,                     -- Primary player ID (FK to dim_players)
    secondary_player_id VARCHAR,           -- Secondary player ID if applicable
    score_home INTEGER,                    -- Home score after play
    score_away INTEGER,                    -- Away score after play
    score_change INTEGER,                  -- Points scored on play
    
    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (team_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (player_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (secondary_player_id) REFERENCES dim_players(player_id)
);
```

##### `fact_team_rankings`
```sql
CREATE TABLE fact_team_rankings (
    ranking_id VARCHAR,                    -- Unique ranking identifier
    ranking_date DATE,                     -- Date of ranking
    season_id INTEGER,                     -- Season ID (FK to dim_seasons)
    poll_name VARCHAR,                     -- Poll name (AP, Coaches, etc.)
    team_id VARCHAR,                       -- Team ID (FK to dim_teams)
    rank INTEGER,                          -- Numerical rank
    points INTEGER,                        -- Poll points
    first_place_votes INTEGER,             -- First place votes
    previous_rank INTEGER,                 -- Previous week's rank
    
    PRIMARY KEY (ranking_id, team_id),
    FOREIGN KEY (season_id) REFERENCES dim_seasons(season_id),
    FOREIGN KEY (team_id) REFERENCES dim_teams(team_id)
);
```

### Gold Layer Schema (Feature Tables)

Feature tables are denormalized and optimized for model training:

#### `features_team_rolling`
```sql
CREATE TABLE features_team_rolling (
    feature_id UUID PRIMARY KEY,           -- Unique feature identifier
    team_id VARCHAR,                       -- Team ID 
    feature_date DATE,                     -- Date feature is valid for
    season_id INTEGER,                     -- Season ID
    
    -- Rolling window size
    window_size INTEGER,                   -- Number of games in window
    
    -- Basic rolling stats
    games_in_window INTEGER,               -- Actual games in window
    wins_in_window INTEGER,                -- Wins in window
    points_avg DOUBLE,                     -- Average points per game
    points_against_avg DOUBLE,             -- Average points against per game
    fg_pct_avg DOUBLE,                     -- Average field goal percentage
    three_pt_pct_avg DOUBLE,               -- Average three-point percentage
    ft_pct_avg DOUBLE,                     -- Average free throw percentage
    rebound_margin_avg DOUBLE,             -- Average rebounding margin
    turnover_margin_avg DOUBLE,            -- Average turnover margin
    
    -- Advanced rolling stats
    offensive_rating_avg DOUBLE,           -- Average offensive rating
    defensive_rating_avg DOUBLE,           -- Average defensive rating
    net_rating_avg DOUBLE,                 -- Average net rating
    pace_avg DOUBLE,                       -- Average pace
    
    -- Trends
    points_trend DOUBLE,                   -- Trend in scoring
    defensive_rating_trend DOUBLE,         -- Trend in defensive rating
    
    -- Strength metrics
    strength_of_schedule DOUBLE,           -- Strength of recent schedule
    
    -- Context
    home_games_pct DOUBLE,                 -- Percentage of home games in window
    days_rest_avg DOUBLE,                  -- Average days of rest between games
    
    -- Feature metadata
    feature_version INTEGER,               -- Version of feature calculation
    created_at TIMESTAMP,                  -- When feature was calculated
    
    FOREIGN KEY (team_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (season_id) REFERENCES dim_seasons(season_id)
);

-- Index for fast lookups
CREATE INDEX idx_features_team_rolling_lookup 
ON features_team_rolling(team_id, feature_date, window_size);
```

#### `features_team_matchup`
```sql
CREATE TABLE features_team_matchup (
    feature_id UUID PRIMARY KEY,           -- Unique feature identifier
    team_id VARCHAR,                       -- Team ID
    opponent_id VARCHAR,                   -- Opponent team ID
    feature_date DATE,                     -- Date feature is valid for
    season_id INTEGER,                     -- Season ID
    
    -- Historical matchup stats
    games_played INTEGER,                  -- Total games between teams
    wins INTEGER,                          -- Wins against opponent
    points_avg DOUBLE,                     -- Average points vs opponent
    points_against_avg DOUBLE,             -- Average points allowed vs opponent
    
    -- Recent matchup stats
    recent_games INTEGER,                  -- Recent games (last 3 seasons)
    recent_wins INTEGER,                   -- Recent wins
    recent_points_avg DOUBLE,              -- Recent average points
    recent_margin_avg DOUBLE,              -- Recent average margin
    
    -- Style matchup
    pace_difference DOUBLE,                -- Difference in pace
    size_advantage DOUBLE,                 -- Size advantage metric
    three_point_advantage DOUBLE,          -- Three-point shooting advantage
    home_court_advantage DOUBLE,           -- Historical home court advantage
    
    -- Feature metadata
    feature_version INTEGER,               -- Version of feature calculation
    created_at TIMESTAMP,                  -- When feature was calculated
    
    FOREIGN KEY (team_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (opponent_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (season_id) REFERENCES dim_seasons(season_id)
);
```

#### `features_game_prediction`
```sql
CREATE TABLE features_game_prediction (
    feature_id UUID PRIMARY KEY,           -- Unique feature identifier
    game_id VARCHAR,                       -- Game ID
    home_team_id VARCHAR,                  -- Home team ID
    away_team_id VARCHAR,                  -- Away team ID
    feature_date DATE,                     -- Date feature is valid for
    
    -- Game context
    is_conference_game BOOLEAN,            -- Whether game is conference game
    is_tournament_game BOOLEAN,            -- Whether game is tournament game
    is_neutral_site BOOLEAN,               -- Whether game is at neutral site
    days_rest_home INTEGER,                -- Days of rest for home team
    days_rest_away INTEGER,                -- Days of rest for away team
    
    -- Team strength features
    home_win_pct_season DOUBLE,            -- Home team season win percentage
    away_win_pct_season DOUBLE,            -- Away team season win percentage
    home_net_rating_season DOUBLE,         -- Home team season net rating
    away_net_rating_season DOUBLE,         -- Away team season net rating
    
    -- Rolling window features (using various windows)
    home_points_avg_10 DOUBLE,             -- Home team avg points (10 games)
    away_points_avg_10 DOUBLE,             -- Away team avg points (10 games)
    home_defensive_rating_10 DOUBLE,       -- Home defensive rating (10 games)
    away_defensive_rating_10 DOUBLE,       -- Away defensive rating (10 games)
    
    -- Matchup features
    historical_matchup_advantage DOUBLE,   -- Historical matchup advantage
    
    -- Ranking features
    home_ap_rank INTEGER,                  -- Home team AP rank
    away_ap_rank INTEGER,                  -- Away team AP rank
    
    -- Vegas line if available
    vegas_spread DOUBLE,                   -- Vegas spread (negative = home favored)
    vegas_total DOUBLE,                    -- Vegas over/under total
    
    -- Feature metadata
    feature_version INTEGER,               -- Version of feature calculation
    created_at TIMESTAMP,                  -- When feature was calculated
    
    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (home_team_id) REFERENCES dim_teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES dim_teams(team_id)
);
```

#### `model_predictions`
```sql
CREATE TABLE model_predictions (
    prediction_id UUID PRIMARY KEY,        -- Unique prediction identifier
    game_id VARCHAR,                       -- Game ID
    model_version VARCHAR,                 -- Model version identifier
    prediction_date TIMESTAMP,             -- When prediction was made
    
    -- Prediction outputs
    predicted_home_win_prob DOUBLE,        -- Probability of home win
    predicted_away_win_prob DOUBLE,        -- Probability of away win
    predicted_spread DOUBLE,               -- Predicted point spread
    predicted_total DOUBLE,                -- Predicted total points
    prediction_confidence DOUBLE,          -- Model confidence (0-1)
    
    -- Actual results (filled after game)
    actual_home_win BOOLEAN,               -- Whether home team won
    actual_spread DOUBLE,                  -- Actual point spread
    actual_total DOUBLE,                   -- Actual total points
    
    -- Prediction evaluation (filled after game)
    prediction_correct BOOLEAN,            -- Whether prediction was correct
    spread_error DOUBLE,                   -- Error in spread prediction
    total_error DOUBLE,                    -- Error in total prediction
    brier_score DOUBLE,                    -- Brier score for probability calibration
    
    created_at TIMESTAMP,                  -- When prediction was created
    updated_at TIMESTAMP,                  -- When prediction was last updated
    
    FOREIGN KEY (game_id) REFERENCES fact_games(game_id)
);
```

## Database Implementation in DuckDB

The tables are implemented in DuckDB using SQL DDL statements. Since DuckDB works directly with Parquet files, we can structure the data storage as follows:

```python
import duckdb

# Connect to database
conn = duckdb.connect('data/ncaa_basketball.duckdb')

# Create schema from DDL files
with open('sql/schema/dim_tables.sql', 'r') as f:
    conn.execute(f.read())
    
with open('sql/schema/fact_tables.sql', 'r') as f:
    conn.execute(f.read())
    
with open('sql/schema/feature_tables.sql', 'r') as f:
    conn.execute(f.read())

# Configure Parquet storage
conn.execute("""
PRAGMA enable_object_cache;
PRAGMA memory_limit='4GB';
PRAGMA threads=4;
"""
)

# Create views for external Parquet files
conn.execute("""
CREATE VIEW IF NOT EXISTS fact_games AS
SELECT * FROM parquet_scan('data/processed/games/*.parquet');

CREATE VIEW IF NOT EXISTS fact_team_game_stats AS
SELECT * FROM parquet_scan('data/processed/team_game_stats/*.parquet');

-- Additional views for other tables
""")
```

## Table Relationships

The relationships between tables follow a star schema pattern:

- **Dimension tables** (dim_*) contain descriptive attributes for entities
- **Fact tables** (fact_*) contain measurements and events
- **Feature tables** (features_*) contain derived data for ML models

Key relationships include:

1. `fact_games` references `dim_teams` for both home and away teams
2. `fact_team_game_stats` references both `fact_games` and `dim_teams`
3. `fact_player_game_stats` references `fact_games`, `dim_players`, and `dim_teams`
4. Feature tables reference dimension tables for entity identification

## Data Management Strategy

### Loading Process

1. **Raw Data Extraction**: ESPN API data is stored as JSON in the bronze layer
2. **Data Validation**: JSON is validated against expected schemas
3. **Transformation**: Cleaned data is converted to Parquet files with normalized schema
4. **Loading**: SQL views and/or tables are created over the Parquet files
5. **Feature Generation**: Features are calculated and stored in the gold layer

### Update Strategy

1. **Incremental Updates**: Most updates are incremental, appending new data
2. **Versioning**: Feature definitions have version tracking
3. **Historical Preservation**: Historical data is preserved for backtesting
4. **Full Recalculation**: Periodic full recalculation ensures consistency

### Partitioning Strategy

Parquet files are partitioned in several ways:

1. **Time-based partitioning**: Data split by season/year
2. **Entity-based partitioning**: Some data split by team or conference
3. **Feature-based partitioning**: Features split by type and window size

## Implementation Considerations

### 1. Data Access Patterns

The schema is optimized for these access patterns:

- Retrieving team performance over time
- Analyzing matchup history between teams
- Generating features for upcoming games
- Evaluating prediction accuracy

### 2. Storage Considerations

- **File Format**: Parquet with Zstandard compression
- **Storage Requirements**: Approximately 50-100GB for 20 years of data
- **Partitioning**: Time-based partitioning for efficient querying

### 3. Incremental Processing

- **Detect Changes**: Track last update timestamp
- **Process New Data**: Only process new or changed records
- **Update Features**: Incrementally update features

## Sample Queries

### Game Results Query

```sql
SELECT
    g.game_id,
    g.game_date,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    g.home_score,
    g.away_score,
    CASE WHEN g.home_score > g.away_score THEN ht.team_name ELSE at.team_name END AS winner
FROM
    fact_games g
JOIN
    dim_teams ht ON g.home_team_id = ht.team_id
JOIN
    dim_teams at ON g.away_team_id = at.team_id
WHERE
    g.season_id = 2024
    AND g.home_score IS NOT NULL
ORDER BY
    g.game_date DESC
LIMIT 10;
```

### Team Performance Query

```sql
SELECT
    t.team_name,
    COUNT(*) AS games_played,
    SUM(CASE WHEN 
        (tgs.is_home AND g.home_score > g.away_score) OR 
        (NOT tgs.is_home AND g.away_score > g.home_score) 
        THEN 1 ELSE 0 END) AS wins,
    AVG(tgs.points) AS avg_points,
    AVG(tgs.rebounds_total) AS avg_rebounds,
    AVG(tgs.assists) AS avg_assists
FROM
    fact_team_game_stats tgs
JOIN
    fact_games g ON tgs.game_id = g.game_id
JOIN
    dim_teams t ON tgs.team_id = t.team_id
WHERE
    g.season_id = 2024
GROUP BY
    t.team_name
ORDER BY
    wins DESC
LIMIT 10;
```

### Feature Generation Query

```sql
SELECT
    team_id,
    feature_date,
    AVG(points) OVER(PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS points_avg_5,
    AVG(offensive_rating) OVER(PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS off_rtg_avg_5,
    AVG(defensive_rating) OVER(PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS def_rtg_avg_5
FROM
    fact_team_game_stats
WHERE
    season_id = 2024
ORDER BY
    team_id,
    game_date;
```

## Migration and Evolution Strategy

The schema is designed to evolve over time:

1. **New Features**: Additional feature tables can be added
2. **Schema Extensions**: New columns can be added to existing tables
3. **Version Control**: Feature definitions include version tracking
4. **Backward Compatibility**: Model inputs maintain consistency

This ensures that as the project evolves, the data structure can adapt while maintaining historical consistency.
