"""
Basic team statistics features.

This module contains features for basic team statistics like wins, losses,
and win percentage.
"""

from typing import Dict, Any
import polars as pl

from src.features.base import CachedFeature
from src.features.registry import FeatureMetadata


class WinsFeature(CachedFeature):
    """Feature that computes total wins for a team."""

    def __init__(self, version: str = "1.0.0"):
        """
        Initialize the wins feature.

        Args:
            version: Version of the feature
        """
        metadata = FeatureMetadata(
            description="Total number of wins for a team",
            category="team_performance",
            tags=["team", "performance", "basic"],
        )
        super().__init__(name="wins", version=version, metadata=metadata)

    def _compute_uncached(self, data: Dict[str, Any]) -> int:
        """
        Compute the total number of wins.

        Args:
            data: Dictionary containing team games data with game results

        Returns:
            Integer count of wins
        """
        if "games" not in data:
            raise ValueError("Team games data not found")

        games = data["games"]

        if isinstance(games, pl.DataFrame):
            # If games is a DataFrame, filter for wins
            team_id = data.get("team_id")
            if not team_id:
                raise ValueError("Team ID not provided")

            # Assuming wins are where team_score > opponent_score
            # Adapt this logic based on your actual data schema
            wins = games.filter(
                (pl.col("team_id") == team_id) & (pl.col("team_score") > pl.col("opponent_score"))
            ).height

            return wins
        elif isinstance(games, list):
            # If games is a list, count wins
            team_id = data.get("team_id")
            if not team_id:
                raise ValueError("Team ID not provided")

            # Adapt this logic based on your actual data structure
            wins = sum(
                1
                for game in games
                if game.get("team_id") == team_id
                and game.get("team_score", 0) > game.get("opponent_score", 0)
            )

            return wins
        else:
            raise ValueError("Unsupported games data format")


class LossesFeature(CachedFeature):
    """Feature that computes total losses for a team."""

    def __init__(self, version: str = "1.0.0"):
        """
        Initialize the losses feature.

        Args:
            version: Version of the feature
        """
        metadata = FeatureMetadata(
            description="Total number of losses for a team",
            category="team_performance",
            tags=["team", "performance", "basic"],
        )
        super().__init__(name="losses", version=version, metadata=metadata)

    def _compute_uncached(self, data: Dict[str, Any]) -> int:
        """
        Compute the total number of losses.

        Args:
            data: Dictionary containing team games data with game results

        Returns:
            Integer count of losses
        """
        if "games" not in data:
            raise ValueError("Team games data not found")

        games = data["games"]

        if isinstance(games, pl.DataFrame):
            # If games is a DataFrame, filter for losses
            team_id = data.get("team_id")
            if not team_id:
                raise ValueError("Team ID not provided")

            # Assuming losses are where team_score < opponent_score
            # Adapt this logic based on your actual data schema
            losses = games.filter(
                (pl.col("team_id") == team_id) & (pl.col("team_score") < pl.col("opponent_score"))
            ).height

            return losses
        elif isinstance(games, list):
            # If games is a list, count losses
            team_id = data.get("team_id")
            if not team_id:
                raise ValueError("Team ID not provided")

            # Adapt this logic based on your actual data structure
            losses = sum(
                1
                for game in games
                if game.get("team_id") == team_id
                and game.get("team_score", 0) < game.get("opponent_score", 0)
            )

            return losses
        else:
            raise ValueError("Unsupported games data format")


class WinPercentageFeature(CachedFeature):
    """Feature that computes win percentage for a team."""

    def __init__(self, version: str = "1.0.0"):
        """
        Initialize the win percentage feature.

        Args:
            version: Version of the feature
        """
        metadata = FeatureMetadata(
            description="Win percentage for a team (wins / games played)",
            category="team_performance",
            tags=["team", "performance", "basic"],
        )
        # Depends on wins and losses features
        dependencies = ["wins", "losses"]
        super().__init__(
            name="win_percentage", version=version, metadata=metadata, dependencies=dependencies
        )

    def _compute_uncached(self, data: Dict[str, Any]) -> float:
        """
        Compute the win percentage.

        Args:
            data: Dictionary containing wins and losses

        Returns:
            Float representing win percentage (0.0 to 1.0)
        """
        wins = data.get("wins")
        losses = data.get("losses")

        if wins is None or losses is None:
            raise ValueError("Wins and losses data required for win percentage calculation")

        games_played = wins + losses

        if games_played == 0:
            return 0.0

        return wins / games_played


class PointsPerGameFeature(CachedFeature):
    """Feature that computes points per game for a team."""

    def __init__(self, version: str = "1.0.0"):
        """
        Initialize the points per game feature.

        Args:
            version: Version of the feature
        """
        metadata = FeatureMetadata(
            description="Average points scored per game by a team",
            category="team_performance",
            tags=["team", "performance", "offense", "basic"],
        )
        super().__init__(name="points_per_game", version=version, metadata=metadata)

    def _compute_uncached(self, data: Dict[str, Any]) -> float:
        """
        Compute the average points per game.

        Args:
            data: Dictionary containing team games data with scores

        Returns:
            Float representing average points per game
        """
        if "games" not in data:
            raise ValueError("Team games data not found")

        games = data["games"]
        team_id = data.get("team_id")

        if not team_id:
            raise ValueError("Team ID not provided")

        if isinstance(games, pl.DataFrame):
            # Filter for games played by this team
            team_games = games.filter(pl.col("team_id") == team_id)

            if team_games.is_empty():
                return 0.0

            # Calculate average points
            avg_points = team_games.select(pl.mean("team_score")).item()
            return float(avg_points)

        elif isinstance(games, list):
            # Calculate from list of games
            team_games = [game for game in games if game.get("team_id") == team_id]

            if not team_games:
                return 0.0

            total_points = sum(game.get("team_score", 0) for game in team_games)
            return total_points / len(team_games)

        else:
            raise ValueError("Unsupported games data format")
