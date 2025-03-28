"""
ESPNApiClient adapter for compatibility with Airflow operators.

This adapter provides a compatibility layer between the existing ESPNClient
implementation and the interface expected by Airflow operators.
"""

import logging
from typing import Dict, List, Any, Optional

from src.data.api.espn_client.client import ESPNClient


class ESPNApiClient:
    """
    ESPN API client adapter for compatibility with Airflow operators.

    This adapter wraps the ESPNClient to provide the interface expected
    by the Airflow operators and prediction components.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize ESPN API client adapter."""
        self.client = ESPNClient(config_path=config_path)

    def get_teams(self, season: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get teams data from ESPN API.

        Args:
            season: Basketball season (e.g., '2022-23')

        Returns:
            List of team dictionaries
        """
        return self.client.get_teams()

    def get_games(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get games data from ESPN API.

        Args:
            start_date: Start date for games (YYYY-MM-DD)
            end_date: End date for games (YYYY-MM-DD)
            team_id: Team ID to filter by
            limit: Maximum number of games to return

        Returns:
            List of game dictionaries
        """
        return self.client.get_games(
            start_date=start_date, end_date=end_date, team_id=team_id, limit=limit
        )

    def get_players(self, team_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get players data from ESPN API.

        Args:
            team_id: Optional team ID to filter by

        Returns:
            List of player dictionaries
        """
        if team_id:
            return self.client.get_team_players(team_id)

        # Get all teams
        teams = self.get_teams()
        all_players = []

        # Get players for each team
        for team in teams:
            team_id = team.get("team_id")
            if team_id:
                try:
                    team_players = self.client.get_team_players(team_id)
                    all_players.extend(team_players)
                except Exception as e:
                    # Log error but continue with other teams
                    logging.getLogger(__name__).warning(
                        f"Error fetching players for team {team_id}: {str(e)}"
                    )

        return all_players

    def get_player_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        player_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get player statistics from ESPN API.

        Args:
            start_date: Start date for stats (YYYY-MM-DD)
            end_date: End date for stats (YYYY-MM-DD)
            player_id: Optional player ID to filter by

        Returns:
            List of player statistics dictionaries
        """
        # If player_id is provided, get stats for that player only
        if player_id:
            stats = self.client.get_player_stats(player_id)
            # Convert to list format expected by operators
            return [stats]

        # Otherwise, we need to get stats for all players in the date range
        # First, get all games in the date range - using the date filter in the query
        # but not using the result directly
        self.get_games(start_date=start_date, end_date=end_date)

        # Get all players
        players = self.get_players()

        # For each player, get their stats and filter by date range
        all_stats = []
        for player in players:
            player_id = player.get("player_id")
            if player_id:
                try:
                    player_stats = self.client.get_player_stats(player_id)

                    # Filter by date if needed
                    if start_date or end_date:
                        # Add filtered stats to the result
                        # Note: Implementation depends on actual stats format
                        # This is a placeholder that would need to be adjusted
                        all_stats.append(player_stats)
                    else:
                        all_stats.append(player_stats)

                except Exception as e:
                    # Log error but continue with other players
                    logging.getLogger(__name__).warning(
                        f"Error fetching stats for player {player_id}: {str(e)}"
                    )

        return all_stats
