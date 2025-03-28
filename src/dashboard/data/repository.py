import duckdb
import polars as pl
import os
from typing import Optional, Any
import time


class DashboardRepository:
    """Repository for accessing data needed by the dashboard."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the repository.

        Args:
            db_path: Path to the DuckDB database file. If None, uses the default path.
        """
        self.db_path = db_path or os.environ.get("DUCKDB_PATH", "data/processed/basketball.db")
        self._connection = None
        self._cache = {}
        self._cache_ttl = {}
        self._default_ttl = 300  # 5 minutes

    @property
    def connection(self):
        """Get or create the database connection."""
        if self._connection is None:
            self._connection = duckdb.connect(self.db_path, read_only=True)
        return self._connection

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get a value from cache if it exists and is not expired."""
        if key in self._cache:
            if self._cache_ttl.get(key, 0) > time.time():
                return self._cache[key]
            else:
                # Clean up expired cache entry
                del self._cache[key]
                if key in self._cache_ttl:
                    del self._cache_ttl[key]
        return None

    def _set_cache(self, key: str, value: Any, ttl: int = None) -> None:
        """Set a value in the cache with a TTL."""
        ttl = ttl or self._default_ttl
        self._cache[key] = value
        self._cache_ttl[key] = time.time() + ttl

    def get_teams(self) -> pl.DataFrame:
        """
        Get all teams.

        Returns:
            DataFrame containing team information
        """
        cache_key = "teams"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # In a real implementation, this would query the database
        # For now, we'll return dummy data
        teams = pl.DataFrame(
            {
                "team_id": ["TEAM1", "TEAM2", "TEAM3"],
                "team_name": ["Team One", "Team Two", "Team Three"],
                "conference": ["Conference A", "Conference B", "Conference A"],
            }
        )

        self._set_cache(cache_key, teams)
        return teams

    def get_recent_games(self, team_id: Optional[str] = None, limit: int = 10) -> pl.DataFrame:
        """
        Get recent games, optionally filtered by team.

        Args:
            team_id: Optional team ID to filter by
            limit: Maximum number of games to return

        Returns:
            DataFrame containing game information
        """
        cache_key = f"recent_games_{team_id}_{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # In a real implementation, this would query the database
        # For now, we'll return dummy data
        games = pl.DataFrame(
            {
                "game_id": [f"GAME{i}" for i in range(1, limit + 1)],
                "game_date": ["2023-02-01", "2023-02-05", "2023-02-10", "2023-02-15", "2023-02-20"],
                "home_team_id": ["TEAM1", "TEAM2", "TEAM3", "TEAM1", "TEAM2"],
                "away_team_id": ["TEAM2", "TEAM3", "TEAM1", "TEAM3", "TEAM1"],
                "home_score": [75, 82, 68, 91, 77],
                "away_score": [70, 75, 72, 85, 80],
            }
        )

        # Filter by team_id if provided
        if team_id:
            games = games.filter(
                (pl.col("home_team_id") == team_id) | (pl.col("away_team_id") == team_id)
            )

        # Limit results
        games = games.head(limit)

        self._set_cache(cache_key, games)
        return games

    def get_players(self, team_id: str) -> pl.DataFrame:
        """
        Get players for a specific team.

        Args:
            team_id: Team ID to get players for

        Returns:
            DataFrame containing player information
        """
        cache_key = f"players_{team_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # In a real implementation, this would query the database
        # For now, we'll return dummy data
        players = pl.DataFrame(
            {
                "player_id": [f"PLAYER{i}" for i in range(1, 6)],
                "player_name": [
                    "John Smith",
                    "Mike Johnson",
                    "Chris Davis",
                    "Sam Brown",
                    "Alex Wilson",
                ],
                "team_id": [team_id] * 5,
                "position": ["G", "F", "C", "G", "F"],
                "jersey_number": [10, 23, 32, 5, 15],
            }
        )

        self._set_cache(cache_key, players)
        return players

    def get_player_stats(self, player_id: str) -> pl.DataFrame:
        """
        Get statistics for a specific player.

        Args:
            player_id: Player ID to get statistics for

        Returns:
            DataFrame containing player statistics
        """
        cache_key = f"player_stats_{player_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # In a real implementation, this would query the database
        # For now, we'll return dummy data
        stats = pl.DataFrame(
            {
                "player_id": [player_id] * 5,
                "game_id": [f"GAME{i}" for i in range(1, 6)],
                "game_date": ["2023-02-01", "2023-02-05", "2023-02-10", "2023-02-15", "2023-02-20"],
                "points": [15, 22, 18, 25, 12],
                "rebounds": [5, 7, 6, 8, 4],
                "assists": [3, 5, 2, 6, 3],
                "steals": [1, 2, 1, 3, 0],
                "blocks": [0, 1, 2, 1, 0],
            }
        )

        self._set_cache(cache_key, stats)
        return stats
