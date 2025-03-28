"""
ESPN API operators for Apache Airflow.
These operators interact with the ESPN API to collect NCAA basketball data.
"""

from typing import Any, Dict, Optional

from airflow.models import BaseOperator
from airflow.hooks.duckdb_hook import DuckDBHook
from airflow.utils.decorators import apply_defaults

# Assuming ESPNApiClient is implemented elsewhere in the project
from src.data.api.espn_client import ESPNApiClient


class BaseESPNOperator(BaseOperator):
    """
    Base operator for ESPN API operations.

    :param conn_id: Connection ID for DuckDB connection
    :param database: Path to the DuckDB database file
    """

    @apply_defaults
    def __init__(self, conn_id: str, database: str, **kwargs):
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.database = database

    def get_hook(self) -> DuckDBHook:
        """Get DuckDB hook instance."""
        return DuckDBHook(conn_id=self.conn_id, database=self.database)

    def get_api_client(self) -> ESPNApiClient:
        """Get ESPN API client instance."""
        return ESPNApiClient()


class FetchTeamsOperator(BaseESPNOperator):
    """
    Operator for fetching team data from ESPN API and storing in DuckDB.

    :param conn_id: Connection ID for DuckDB connection
    :param database: Path to the DuckDB database file
    :param season: Basketball season (e.g., '2022-23')
    :param create_table: Whether to create the table if it doesn't exist
    """

    template_fields = ("season",)

    @apply_defaults
    def __init__(
        self,
        conn_id: str,
        database: str,
        season: Optional[str] = None,
        create_table: bool = True,
        **kwargs,
    ):
        super().__init__(conn_id=conn_id, database=database, **kwargs)
        self.season = season
        self.create_table = create_table

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute the operator to fetch team data.

        :param context: Airflow context
        """
        # Get API client and database hook
        api_client = self.get_api_client()
        hook = self.get_hook()

        # Fetch teams data
        self.log.info(f"Fetching teams data for season: {self.season}")
        teams = api_client.get_teams(season=self.season)
        self.log.info(f"Retrieved {len(teams)} teams")

        # Create teams table if needed
        if self.create_table:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS teams (
                team_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                abbreviation VARCHAR,
                conference VARCHAR,
                logo_url VARCHAR,
                season VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            hook.run_query(create_table_sql)

        # Insert teams data
        for team in teams:
            insert_sql = """
            INSERT OR REPLACE INTO teams (team_id, name, abbreviation, conference, logo_url, season)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            params = (
                team.get("team_id"),
                team.get("name"),
                team.get("abbreviation"),
                team.get("conference"),
                team.get("logo_url"),
                self.season,
            )
            hook.run_query(insert_sql, dict(zip(range(1, len(params) + 1), params)))

        self.log.info(f"Successfully stored {len(teams)} teams in database")


class FetchGamesOperator(BaseESPNOperator):
    """
    Operator for fetching game data from ESPN API and storing in DuckDB.

    :param conn_id: Connection ID for DuckDB connection
    :param database: Path to the DuckDB database file
    :param start_date: Start date for games to fetch (YYYY-MM-DD)
    :param end_date: End date for games to fetch (YYYY-MM-DD)
    :param create_table: Whether to create the table if it doesn't exist
    :param incremental: Whether to only fetch games not already in the database
    """

    template_fields = ("start_date", "end_date")

    @apply_defaults
    def __init__(
        self,
        conn_id: str,
        database: str,
        start_date: str,
        end_date: str,
        create_table: bool = True,
        incremental: bool = False,
        **kwargs,
    ):
        super().__init__(conn_id=conn_id, database=database, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        self.create_table = create_table
        self.incremental = incremental

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute the operator to fetch game data.

        :param context: Airflow context
        """
        # Get API client and database hook
        api_client = self.get_api_client()
        hook = self.get_hook()

        # Create games table if needed
        if self.create_table:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS games (
                game_id VARCHAR PRIMARY KEY,
                home_team VARCHAR NOT NULL,
                away_team VARCHAR NOT NULL,
                date DATE NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                status VARCHAR,
                venue VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            hook.run_query(create_table_sql)

        # Get existing game IDs for incremental loading
        existing_game_ids = set()
        if self.incremental:
            self.log.info("Fetching existing game IDs for incremental loading")
            existing_games_sql = "SELECT game_id FROM games"
            for record in hook.get_records(existing_games_sql):
                existing_game_ids.add(record[0])

        # Fetch games data
        self.log.info(f"Fetching games from {self.start_date} to {self.end_date}")
        games = api_client.get_games(start_date=self.start_date, end_date=self.end_date)
        self.log.info(f"Retrieved {len(games)} games")

        # Filter out existing games if incremental
        if self.incremental:
            new_games = [game for game in games if game.get("game_id") not in existing_game_ids]
            self.log.info(f"Filtered to {len(new_games)} new games")
            games = new_games

        # Insert games data
        for game in games:
            insert_sql = """
            INSERT OR REPLACE INTO games (
                game_id, home_team, away_team, date, home_score, away_score, status, venue
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                game.get("game_id"),
                game.get("home_team"),
                game.get("away_team"),
                game.get("date"),
                game.get("home_score"),
                game.get("away_score"),
                game.get("status"),
                game.get("venue"),
            )
            hook.run_query(insert_sql, dict(zip(range(1, len(params) + 1), params)))

        self.log.info(f"Successfully stored {len(games)} games in database")


class FetchPlayersOperator(BaseESPNOperator):
    """
    Operator for fetching player data from ESPN API and storing in DuckDB.

    :param conn_id: Connection ID for DuckDB connection
    :param database: Path to the DuckDB database file
    :param season: Basketball season (e.g., '2022-23')
    :param create_table: Whether to create the table if it doesn't exist
    """

    template_fields = ("season",)

    @apply_defaults
    def __init__(
        self,
        conn_id: str,
        database: str,
        season: Optional[str] = None,
        create_table: bool = True,
        **kwargs,
    ):
        super().__init__(conn_id=conn_id, database=database, **kwargs)
        self.season = season
        self.create_table = create_table

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute the operator to fetch player data.

        :param context: Airflow context
        """
        # Get API client and database hook
        api_client = self.get_api_client()
        hook = self.get_hook()

        # Create players table if needed
        if self.create_table:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS players (
                player_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                team_id VARCHAR NOT NULL,
                position VARCHAR,
                jersey VARCHAR,
                height VARCHAR,
                weight VARCHAR,
                season VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            hook.run_query(create_table_sql)

        # Fetch players data
        self.log.info(f"Fetching players data for season: {self.season}")
        players = api_client.get_players(season=self.season)
        self.log.info(f"Retrieved {len(players)} players")

        # Insert players data
        for player in players:
            insert_sql = """
            INSERT OR REPLACE INTO players (
                player_id, name, team_id, position, jersey, height, weight, season
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                player.get("player_id"),
                player.get("name"),
                player.get("team_id"),
                player.get("position"),
                player.get("jersey"),
                player.get("height"),
                player.get("weight"),
                self.season,
            )
            hook.run_query(insert_sql, dict(zip(range(1, len(params) + 1), params)))

        self.log.info(f"Successfully stored {len(players)} players in database")


class FetchPlayerStatsOperator(BaseESPNOperator):
    """
    Operator for fetching player stats from ESPN API and storing in DuckDB.

    :param conn_id: Connection ID for DuckDB connection
    :param database: Path to the DuckDB database file
    :param start_date: Start date for stats to fetch (YYYY-MM-DD)
    :param end_date: End date for stats to fetch (YYYY-MM-DD)
    :param create_table: Whether to create the table if it doesn't exist
    """

    template_fields = ("start_date", "end_date")

    @apply_defaults
    def __init__(
        self,
        conn_id: str,
        database: str,
        start_date: str,
        end_date: str,
        create_table: bool = True,
        **kwargs,
    ):
        super().__init__(conn_id=conn_id, database=database, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        self.create_table = create_table

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute the operator to fetch player stats.

        :param context: Airflow context
        """
        # Get API client and database hook
        api_client = self.get_api_client()
        hook = self.get_hook()

        # Create player stats table if needed
        if self.create_table:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS player_stats (
                player_id VARCHAR,
                game_id VARCHAR,
                minutes INTEGER,
                points INTEGER,
                rebounds INTEGER,
                assists INTEGER,
                steals INTEGER,
                blocks INTEGER,
                turnovers INTEGER,
                field_goals_made INTEGER,
                field_goals_attempted INTEGER,
                three_pointers_made INTEGER,
                three_pointers_attempted INTEGER,
                free_throws_made INTEGER,
                free_throws_attempted INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, game_id)
            )
            """
            hook.run_query(create_table_sql)

        # Fetch player stats
        self.log.info(f"Fetching player stats from {self.start_date} to {self.end_date}")
        player_stats = api_client.get_player_stats(
            start_date=self.start_date, end_date=self.end_date
        )
        self.log.info(f"Retrieved stats for {len(player_stats)} player-game combinations")

        # Insert player stats
        for stat in player_stats:
            insert_sql = """
            INSERT OR REPLACE INTO player_stats (
                player_id, game_id, minutes, points, rebounds, assists, steals, blocks,
                turnovers, field_goals_made, field_goals_attempted, three_pointers_made,
                three_pointers_attempted, free_throws_made, free_throws_attempted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                stat.get("player_id"),
                stat.get("game_id"),
                stat.get("minutes"),
                stat.get("points"),
                stat.get("rebounds"),
                stat.get("assists"),
                stat.get("steals"),
                stat.get("blocks"),
                stat.get("turnovers"),
                stat.get("field_goals_made"),
                stat.get("field_goals_attempted"),
                stat.get("three_pointers_made"),
                stat.get("three_pointers_attempted"),
                stat.get("free_throws_made"),
                stat.get("free_throws_attempted"),
            )
            hook.run_query(insert_sql, dict(zip(range(1, len(params) + 1), params)))

        self.log.info(f"Successfully stored stats for {len(player_stats)} player-game combinations")
