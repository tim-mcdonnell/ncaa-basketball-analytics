"""
Database schema definitions for NCAA Basketball Analytics.

This module defines the schemas for all tables in the database,
following a medallion architecture with raw, dimension, and fact layers.
"""

from typing import Dict


def get_schema_definitions() -> Dict[str, str]:
    """
    Get all database schema definitions.

    Returns:
        Dictionary mapping table names to SQL CREATE statements
    """
    # Raw Tables
    raw_teams_schema = """
    CREATE TABLE IF NOT EXISTS raw_teams (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        team_id VARCHAR NOT NULL UNIQUE,
        raw_data JSON NOT NULL,
        source_url VARCHAR NOT NULL,
        collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        processing_version VARCHAR NOT NULL
    );
    """

    raw_games_schema = """
    CREATE TABLE IF NOT EXISTS raw_games (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        game_id VARCHAR NOT NULL UNIQUE,
        raw_data JSON NOT NULL,
        source_url VARCHAR NOT NULL,
        collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        processing_version VARCHAR NOT NULL
    );
    """

    raw_players_schema = """
    CREATE TABLE IF NOT EXISTS raw_players (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        player_id VARCHAR NOT NULL UNIQUE,
        raw_data JSON NOT NULL,
        source_url VARCHAR NOT NULL,
        collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        processing_version VARCHAR NOT NULL
    );
    """

    # Dimension Tables
    dim_teams_schema = """
    CREATE TABLE IF NOT EXISTS dim_teams (
        team_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        conference VARCHAR,
        division VARCHAR,
        logo_url VARCHAR,
        mascot VARCHAR,
        primary_color VARCHAR,
        secondary_color VARCHAR,
        venue_name VARCHAR,
        venue_capacity INTEGER,
        city VARCHAR,
        state VARCHAR,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """

    dim_players_schema = """
    CREATE TABLE IF NOT EXISTS dim_players (
        player_id VARCHAR PRIMARY KEY,
        team_id VARCHAR NOT NULL,
        first_name VARCHAR NOT NULL,
        last_name VARCHAR NOT NULL,
        jersey_number VARCHAR,
        position VARCHAR,
        height_cm INTEGER,
        weight_kg FLOAT,
        birth_date DATE,
        class_year VARCHAR,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (team_id) REFERENCES dim_teams(team_id)
    );
    """

    dim_seasons_schema = """
    CREATE TABLE IF NOT EXISTS dim_seasons (
        season_id INTEGER PRIMARY KEY,
        year INTEGER NOT NULL,
        type VARCHAR NOT NULL,
        start_date DATE,
        end_date DATE,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (year, type)
    );
    """

    # Fact Tables
    fact_games_schema = """
    CREATE TABLE IF NOT EXISTS fact_games (
        game_id VARCHAR PRIMARY KEY,
        season_id INTEGER NOT NULL,
        home_team_id VARCHAR NOT NULL,
        away_team_id VARCHAR NOT NULL,
        home_score INTEGER,
        away_score INTEGER,
        game_date DATE NOT NULL,
        venue VARCHAR,
        attendance INTEGER,
        status VARCHAR NOT NULL,
        period_count INTEGER,
        overtime_count INTEGER DEFAULT 0,
        neutral_site BOOLEAN DEFAULT FALSE,
        conference_game BOOLEAN DEFAULT FALSE,
        tournament_game BOOLEAN DEFAULT FALSE,
        tournament_name VARCHAR,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (season_id) REFERENCES dim_seasons(season_id),
        FOREIGN KEY (home_team_id) REFERENCES dim_teams(team_id),
        FOREIGN KEY (away_team_id) REFERENCES dim_teams(team_id)
    );
    """

    fact_player_stats_schema = """
    CREATE TABLE IF NOT EXISTS fact_player_stats (
        stat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        game_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        minutes_played INTEGER,
        field_goals_made INTEGER DEFAULT 0,
        field_goals_attempted INTEGER DEFAULT 0,
        three_pointers_made INTEGER DEFAULT 0,
        three_pointers_attempted INTEGER DEFAULT 0,
        free_throws_made INTEGER DEFAULT 0,
        free_throws_attempted INTEGER DEFAULT 0,
        offensive_rebounds INTEGER DEFAULT 0,
        defensive_rebounds INTEGER DEFAULT 0,
        total_rebounds INTEGER DEFAULT 0,
        assists INTEGER DEFAULT 0,
        steals INTEGER DEFAULT 0,
        blocks INTEGER DEFAULT 0,
        turnovers INTEGER DEFAULT 0,
        personal_fouls INTEGER DEFAULT 0,
        points INTEGER DEFAULT 0,
        plus_minus INTEGER,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
        FOREIGN KEY (player_id) REFERENCES dim_players(player_id),
        FOREIGN KEY (team_id) REFERENCES dim_teams(team_id),
        UNIQUE (game_id, player_id)
    );
    """

    # Combine all schemas into a dictionary
    return {
        "raw_teams": raw_teams_schema,
        "raw_games": raw_games_schema,
        "raw_players": raw_players_schema,
        "dim_teams": dim_teams_schema,
        "dim_players": dim_players_schema,
        "dim_seasons": dim_seasons_schema,
        "fact_games": fact_games_schema,
        "fact_player_stats": fact_player_stats_schema,
    }
