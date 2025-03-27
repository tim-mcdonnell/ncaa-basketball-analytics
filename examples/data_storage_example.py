"""
NCAA Basketball Analytics - Data Storage Example

This script demonstrates the basic usage of the data storage component with DuckDB.
It shows how to initialize a database, define schemas, and use repositories to store and retrieve data.

NOTE: This is a standalone example that doesn't require importing from the project.
"""

import os
import duckdb
from typing import Dict, List, Optional, Any
import uuid
import json
import polars as pl

# Create database in the examples directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "example.duckdb")

# Remove existing database if it exists
if os.path.exists(db_path):
    os.unlink(db_path)

print(f"Using database at: {db_path}")


# ==================== Database Connection ====================


class DBConnection:
    """Database connection manager."""

    def __init__(self, db_path: str):
        """
        Initialize database connection.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get a DuckDB connection.

        Returns:
            DuckDB connection
        """
        return duckdb.connect(self.db_path)

    def execute_script(self, script: str) -> None:
        """
        Execute a SQL script.

        Args:
            script: SQL script to execute
        """
        conn = self.get_connection()
        conn.execute(script)
        conn.close()

    def query_to_polars(self, query: str, params: Optional[tuple] = None) -> pl.DataFrame:
        """
        Execute a query and return results as a Polars DataFrame.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Polars DataFrame with query results
        """
        conn = self.get_connection()
        if params:
            result = conn.execute(query, params).pl()
        else:
            result = conn.execute(query).pl()
        conn.close()
        return result


# ==================== Schema Definitions ====================

SCHEMA_VERSION = "1.0.0"

RAW_TABLES_SCHEMA = """
-- Raw data tables to store data as received from the API

-- Raw Teams Table
CREATE TABLE IF NOT EXISTS raw_teams (
    id UUID PRIMARY KEY,
    team_id VARCHAR NOT NULL,
    raw_data JSON NOT NULL,
    source_url VARCHAR NOT NULL,
    collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_teams_team_id ON raw_teams (team_id);

-- Raw Games Table
CREATE TABLE IF NOT EXISTS raw_games (
    id UUID PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    raw_data JSON NOT NULL,
    source_url VARCHAR NOT NULL,
    collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_games_game_id ON raw_games (game_id);

-- Raw Players Table
CREATE TABLE IF NOT EXISTS raw_players (
    id UUID PRIMARY KEY,
    player_id VARCHAR NOT NULL,
    team_id VARCHAR NOT NULL,
    raw_data JSON NOT NULL,
    source_url VARCHAR NOT NULL,
    collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_players_player_id ON raw_players (player_id);
CREATE INDEX IF NOT EXISTS idx_raw_players_team_id ON raw_players (team_id);
"""

DIMENSIONAL_TABLES_SCHEMA = """
-- Dimensional model tables for analytics

-- Dimension: Teams
CREATE TABLE IF NOT EXISTS dim_teams (
    team_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    short_name VARCHAR,
    abbreviation VARCHAR,
    location VARCHAR,
    conference VARCHAR,
    division VARCHAR,
    logo_url VARCHAR,
    primary_color VARCHAR,
    secondary_color VARCHAR,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Players
CREATE TABLE IF NOT EXISTS dim_players (
    player_id VARCHAR PRIMARY KEY,
    team_id VARCHAR NOT NULL,
    first_name VARCHAR NOT NULL,
    last_name VARCHAR NOT NULL,
    full_name VARCHAR NOT NULL,
    position VARCHAR,
    jersey_number VARCHAR,
    height_inches INTEGER,
    weight_pounds INTEGER,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES dim_teams (team_id)
);

-- Dimension: Seasons
CREATE TABLE IF NOT EXISTS dim_seasons (
    season_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    type VARCHAR NOT NULL
);

-- Fact: Games
CREATE TABLE IF NOT EXISTS fact_games (
    game_id VARCHAR PRIMARY KEY,
    season_id VARCHAR NOT NULL,
    home_team_id VARCHAR NOT NULL,
    away_team_id VARCHAR NOT NULL,
    game_date TIMESTAMP NOT NULL,
    location VARCHAR,
    venue VARCHAR,
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (season_id) REFERENCES dim_seasons (season_id),
    FOREIGN KEY (home_team_id) REFERENCES dim_teams (team_id),
    FOREIGN KEY (away_team_id) REFERENCES dim_teams (team_id)
);

-- Fact: Player Game Stats
CREATE TABLE IF NOT EXISTS fact_player_game_stats (
    id UUID PRIMARY KEY,
    game_id VARCHAR NOT NULL,
    player_id VARCHAR NOT NULL,
    team_id VARCHAR NOT NULL,
    minutes INTEGER,
    points INTEGER,
    field_goals_made INTEGER,
    field_goals_attempted INTEGER,
    three_pointers_made INTEGER,
    three_pointers_attempted INTEGER,
    free_throws_made INTEGER,
    free_throws_attempted INTEGER,
    rebounds_offensive INTEGER,
    rebounds_defensive INTEGER,
    rebounds_total INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fouls INTEGER,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES fact_games (game_id),
    FOREIGN KEY (player_id) REFERENCES dim_players (player_id),
    FOREIGN KEY (team_id) REFERENCES dim_teams (team_id)
);
"""

# ==================== Repository Classes ====================


class BaseRepository:
    """Base repository with common functionality."""

    def __init__(self, db_connection: DBConnection):
        """
        Initialize repository.

        Args:
            db_connection: Database connection
        """
        self.db = db_connection


class RawTeamRepository(BaseRepository):
    """Repository for raw team data."""

    def insert(
        self,
        team_id: str,
        raw_data: Dict[str, Any],
        source_url: str,
        processing_version: str = SCHEMA_VERSION,
    ) -> str:
        """
        Insert raw team data.

        Args:
            team_id: Team ID
            raw_data: Raw team data (will be stored as JSON)
            source_url: Source URL for data
            processing_version: Version of processing code

        Returns:
            UUID of inserted record
        """
        record_id = str(uuid.uuid4())

        conn = self.db.get_connection()
        conn.execute(
            """
            INSERT INTO raw_teams (id, team_id, raw_data, source_url, processing_version)
            VALUES (?, ?, ?, ?, ?)
        """,
            (record_id, team_id, json.dumps(raw_data), source_url, processing_version),
        )
        conn.close()

        return record_id

    def get_by_team_id(self, team_id: str) -> Optional[Dict[str, Any]]:
        """
        Get raw team data by team ID.

        Args:
            team_id: Team ID

        Returns:
            Raw team data or None if not found
        """
        conn = self.db.get_connection()
        result = conn.execute(
            """
            SELECT id, team_id, raw_data, source_url, collected_at, processing_version
            FROM raw_teams
            WHERE team_id = ?
            ORDER BY collected_at DESC
            LIMIT 1
        """,
            (team_id,),
        ).fetchone()
        conn.close()

        if result is None:
            return None

        raw_data = json.loads(result[2])

        return {
            "id": result[0],
            "team_id": result[1],
            "raw_data": raw_data,
            "source_url": result[3],
            "collected_at": result[4],
            "processing_version": result[5],
        }


class DimTeamRepository(BaseRepository):
    """Repository for dimensional team data."""

    def insert_or_update(self, team_data: Dict[str, Any]) -> str:
        """
        Insert or update team in dimensional model.

        Args:
            team_data: Team data

        Returns:
            Team ID
        """
        team_id = team_data["team_id"]

        # Check if team exists
        conn = self.db.get_connection()
        existing = conn.execute(
            """
            SELECT team_id FROM dim_teams WHERE team_id = ?
        """,
            (team_id,),
        ).fetchone()

        if existing:
            # Update existing team
            conn.execute(
                """
                UPDATE dim_teams
                SET
                    name = ?,
                    short_name = ?,
                    abbreviation = ?,
                    location = ?,
                    conference = ?,
                    division = ?,
                    logo_url = ?,
                    primary_color = ?,
                    secondary_color = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE team_id = ?
            """,
                (
                    team_data.get("name", ""),
                    team_data.get("short_name"),
                    team_data.get("abbreviation"),
                    team_data.get("location"),
                    team_data.get("conference"),
                    team_data.get("division"),
                    team_data.get("logo_url"),
                    team_data.get("primary_color"),
                    team_data.get("secondary_color"),
                    team_id,
                ),
            )
        else:
            # Insert new team
            conn.execute(
                """
                INSERT INTO dim_teams (
                    team_id, name, short_name, abbreviation, location,
                    conference, division, logo_url, primary_color, secondary_color
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    team_id,
                    team_data.get("name", ""),
                    team_data.get("short_name"),
                    team_data.get("abbreviation"),
                    team_data.get("location"),
                    team_data.get("conference"),
                    team_data.get("division"),
                    team_data.get("logo_url"),
                    team_data.get("primary_color"),
                    team_data.get("secondary_color"),
                ),
            )

        conn.close()
        return team_id

    def get_by_id(self, team_id: str) -> Optional[Dict[str, Any]]:
        """
        Get team by ID.

        Args:
            team_id: Team ID

        Returns:
            Team data or None if not found
        """
        conn = self.db.get_connection()
        result = conn.execute(
            """
            SELECT * FROM dim_teams WHERE team_id = ?
        """,
            (team_id,),
        ).fetchone()
        conn.close()

        if result is None:
            return None

        # Convert to dictionary
        columns = [
            "team_id",
            "name",
            "short_name",
            "abbreviation",
            "location",
            "conference",
            "division",
            "logo_url",
            "primary_color",
            "secondary_color",
            "updated_at",
        ]

        return {columns[i]: result[i] for i in range(len(result))}

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all teams.

        Returns:
            List of team data
        """
        df = self.db.query_to_polars("""
            SELECT * FROM dim_teams ORDER BY name
        """)

        return df.to_dicts()


# ==================== Data Transformation ====================


def transform_raw_team_to_dim(raw_team: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw team data to dimensional model format.

    Args:
        raw_team: Raw team data

    Returns:
        Transformed team data for dimensional model
    """
    team_data = raw_team["raw_data"]

    # Handle different API response formats
    if "team" in team_data:
        team = team_data["team"]
    else:
        team = team_data

    # Extract team information
    name = team.get("displayName", "")
    location = team.get("location", "")
    abbreviation = team.get("abbreviation", "")

    # Extract conference
    conference = None
    if "conference" in team:
        conference = team["conference"].get("name", "")

    # Extract colors
    primary_color = None
    secondary_color = None
    if "color" in team:
        colors = team["color"].split(",")
        primary_color = colors[0] if len(colors) > 0 else None
        secondary_color = colors[1] if len(colors) > 1 else None

    # Construct the dimensional model data
    return {
        "team_id": raw_team["team_id"],
        "name": name,
        "short_name": team.get("shortDisplayName"),
        "abbreviation": abbreviation,
        "location": location,
        "conference": conference,
        "division": team.get("division"),
        "logo_url": team.get("logos", [{}])[0].get("href")
        if "logos" in team and team["logos"]
        else None,
        "primary_color": primary_color,
        "secondary_color": secondary_color,
    }


# ==================== Example Data ====================


def create_sample_data() -> Dict[str, Any]:
    """
    Create sample data for demonstration.

    Returns:
        Dictionary of sample data
    """
    # Sample teams
    michigan = {
        "team_id": "130",
        "raw_data": {
            "team": {
                "id": "130",
                "displayName": "Michigan Wolverines",
                "shortDisplayName": "Michigan",
                "abbreviation": "MICH",
                "location": "Michigan",
                "color": "00274c,ffcb05",
                "conference": {"id": "5", "name": "Big Ten"},
                "logos": [{"href": "https://a.espncdn.com/i/teamlogos/ncaa/500/130.png"}],
            }
        },
        "source_url": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/130",
    }

    msu = {
        "team_id": "127",
        "raw_data": {
            "team": {
                "id": "127",
                "displayName": "Michigan State Spartans",
                "shortDisplayName": "Michigan State",
                "abbreviation": "MSU",
                "location": "Michigan State",
                "color": "18453b,ffffff",
                "conference": {"id": "5", "name": "Big Ten"},
                "logos": [{"href": "https://a.espncdn.com/i/teamlogos/ncaa/500/127.png"}],
            }
        },
        "source_url": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/127",
    }

    # Sample season
    season_2023 = {
        "season_id": "2023",
        "name": "2022-23 NCAA Basketball",
        "start_date": "2022-11-01",
        "end_date": "2023-04-30",
        "type": "regular",
    }

    # Sample game
    michigan_vs_msu = {
        "game_id": "401458136",
        "season_id": "2023",
        "home_team_id": "130",
        "away_team_id": "127",
        "game_date": "2023-02-18T17:30:00Z",
        "location": "Ann Arbor, MI",
        "venue": "Crisler Center",
        "home_score": 84,
        "away_score": 72,
        "status": "final",
    }

    return {"teams": [michigan, msu], "seasons": [season_2023], "games": [michigan_vs_msu]}


# ==================== Main Example ====================


def main():
    """Demonstrate data storage usage."""
    print("NCAA Basketball Analytics - Data Storage Example\n")

    # Create database connection
    db = DBConnection(db_path)

    # Step 1: Initialize database schema
    print("Step 1: Initializing database schema...")
    db.execute_script(RAW_TABLES_SCHEMA)
    db.execute_script(DIMENSIONAL_TABLES_SCHEMA)
    print("Database schema created successfully.\n")

    # Step 2: Create repositories
    raw_team_repo = RawTeamRepository(db)
    dim_team_repo = DimTeamRepository(db)

    # Step 3: Get sample data
    print("Step 3: Creating sample data...")
    sample_data = create_sample_data()
    print(
        f"Created sample data with {len(sample_data['teams'])} teams, "
        f"{len(sample_data['seasons'])} seasons, {len(sample_data['games'])} games.\n"
    )

    # Step 4: Store raw team data
    print("Step 4: Storing raw team data...")
    for team in sample_data["teams"]:
        record_id = raw_team_repo.insert(
            team_id=team["team_id"], raw_data=team["raw_data"], source_url=team["source_url"]
        )
        print(f"Stored raw team data for {team['team_id']} with ID: {record_id}")
    print()

    # Step 5: Transform and store dimensional data
    print("Step 5: Transforming and storing dimensional team data...")
    for team in sample_data["teams"]:
        # Get raw data from database
        raw_team = raw_team_repo.get_by_team_id(team["team_id"])

        # Transform to dimensional model
        dim_team = transform_raw_team_to_dim(raw_team)

        # Store in dimensional model
        team_id = dim_team_repo.insert_or_update(dim_team)
        print(f"Stored dimensional team data for {team_id}")
    print()

    # Step 6: Query the dimensional model
    print("Step 6: Querying dimensional team data...")
    all_teams = dim_team_repo.get_all()

    print(f"Found {len(all_teams)} teams in dimensional model:")
    for i, team in enumerate(all_teams, 1):
        print(f"{i}. {team['name']} ({team['team_id']})")
        print(f"   Conference: {team['conference']}")
        print(f"   Colors: {team['primary_color']}, {team['secondary_color']}")
        print()

    # Step 7: Run SQL query directly using Polars
    print("Step 7: Running SQL query directly using Polars...")
    query = """
        SELECT
            t.name AS team_name,
            t.conference,
            t.location
        FROM
            dim_teams t
        ORDER BY
            t.conference, t.name
    """

    results = db.query_to_polars(query)
    print("Teams by conference:")
    print(results)
    print()

    # Cleanup
    print(f"Example complete. The database is at: {db_path}")
    print("You can delete it manually when done exploring.")


if __name__ == "__main__":
    main()
