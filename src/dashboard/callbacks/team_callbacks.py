from dash import Dash, Input, Output, html, dash_table
import plotly.graph_objects as go
import polars as pl

from ..components.selectors import TeamSelector
from ..data.repository import DashboardRepository
from ..figures.team_figures import create_team_performance_chart


def register_team_callbacks(app: Dash) -> None:
    """
    Register callbacks for the team analysis page.

    Args:
        app: The Dash application instance
    """
    repository = DashboardRepository()

    # Team selector callback
    @app.callback(
        Output("team-selector-container", "children"),
        Input("_", "children"),  # Dummy input for initialization
    )
    def populate_team_selector(_):
        """Populate the team selector with teams from the database."""
        teams = repository.get_teams()
        team_options = [
            {"label": row["team_name"], "value": row["team_id"]} for row in teams.to_dicts()
        ]

        return TeamSelector(
            id="team-selector",
            teams=team_options,
            value=team_options[0]["value"] if team_options else None,
        )

    # Team performance chart callback
    @app.callback(
        Output("team-performance-chart", "figure"),
        Input("team-selector", "value"),
    )
    def update_team_performance_chart(team_id):
        """Update the team performance chart based on the selected team."""
        if not team_id:
            return go.Figure()

        # Get team name
        teams = repository.get_teams()
        team_name = teams.filter(pl.col("team_id") == team_id)
        team_name = team_name[0, "team_name"] if len(team_name) > 0 else "Unknown Team"

        # Get recent games for the team
        games = repository.get_recent_games(team_id=team_id, limit=10)

        # Create a new DataFrame with the required columns
        performance_data = pl.DataFrame(
            {
                "game_date": games["game_date"],
                "points_scored": [
                    row["home_score"] if row["home_team_id"] == team_id else row["away_score"]
                    for row in games.to_dicts()
                ],
                "points_allowed": [
                    row["away_score"] if row["home_team_id"] == team_id else row["home_score"]
                    for row in games.to_dicts()
                ],
                "win": [
                    1
                    if (row["home_team_id"] == team_id and row["home_score"] > row["away_score"])
                    or (row["away_team_id"] == team_id and row["away_score"] > row["home_score"])
                    else 0
                    for row in games.to_dicts()
                ],
            }
        )

        # Create the chart
        return create_team_performance_chart(performance_data, team_name=team_name)

    # Team stats table callback
    @app.callback(
        Output("team-stats-table", "children"),
        Input("team-selector", "value"),
    )
    def update_team_stats_table(team_id):
        """Update the team statistics table based on the selected team."""
        if not team_id:
            return html.Div("Select a team to view statistics")

        # Get team name
        teams = repository.get_teams()
        team_name = teams.filter(pl.col("team_id") == team_id)
        team_name = team_name[0, "team_name"] if len(team_name) > 0 else "Unknown Team"

        # Get recent games for the team
        games = repository.get_recent_games(team_id=team_id, limit=10)

        # Calculate summary statistics
        if len(games) == 0:
            return html.Div("No games found for the selected team")

        # Determine points scored and allowed for each game
        points_scored = [
            row["home_score"] if row["home_team_id"] == team_id else row["away_score"]
            for row in games.to_dicts()
        ]

        points_allowed = [
            row["away_score"] if row["home_team_id"] == team_id else row["home_score"]
            for row in games.to_dicts()
        ]

        wins = sum(
            [
                1
                if (row["home_team_id"] == team_id and row["home_score"] > row["away_score"])
                or (row["away_team_id"] == team_id and row["away_score"] > row["home_score"])
                else 0
                for row in games.to_dicts()
            ]
        )

        # Calculate statistics
        avg_points_scored = sum(points_scored) / len(points_scored) if points_scored else 0
        avg_points_allowed = sum(points_allowed) / len(points_allowed) if points_allowed else 0
        win_percentage = wins / len(games) * 100 if games else 0

        # Create statistics data
        stats_data = [
            {"Statistic": "Games Played", "Value": len(games)},
            {"Statistic": "Wins", "Value": wins},
            {"Statistic": "Losses", "Value": len(games) - wins},
            {"Statistic": "Win Percentage", "Value": f"{win_percentage:.1f}%"},
            {"Statistic": "Average Points Scored", "Value": f"{avg_points_scored:.1f}"},
            {"Statistic": "Average Points Allowed", "Value": f"{avg_points_allowed:.1f}"},
        ]

        # Create table
        return dash_table.DataTable(
            data=stats_data,
            columns=[
                {"name": "Statistic", "id": "Statistic"},
                {"name": "Value", "id": "Value"},
            ],
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_cell={
                "textAlign": "left",
                "padding": "15px",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "rgb(248, 248, 248)",
                }
            ],
        )

    # Recent games table callback
    @app.callback(
        Output("recent-games-table", "children"),
        Input("team-selector", "value"),
    )
    def update_recent_games_table(team_id):
        """Update the recent games table based on the selected team."""
        if not team_id:
            return html.Div("Select a team to view recent games")

        # Get recent games for the team
        games = repository.get_recent_games(team_id=team_id, limit=5)

        if len(games) == 0:
            return html.Div("No recent games found for the selected team")

        # Create a list of game results
        game_results = []
        for game in games.to_dicts():
            is_home = game["home_team_id"] == team_id
            opponent_id = game["away_team_id"] if is_home else game["home_team_id"]

            # Get opponent name
            teams = repository.get_teams()
            opponent_name = teams.filter(pl.col("team_id") == opponent_id)
            opponent_name = (
                opponent_name[0, "team_name"] if len(opponent_name) > 0 else "Unknown Team"
            )

            team_score = game["home_score"] if is_home else game["away_score"]
            opponent_score = game["away_score"] if is_home else game["home_score"]

            result = "W" if team_score > opponent_score else "L"

            game_results.append(
                {
                    "Date": game["game_date"],
                    "Opponent": opponent_name,
                    "Score": f"{team_score} - {opponent_score}",
                    "Result": result,
                    "Location": "Home" if is_home else "Away",
                }
            )

        # Create table
        return dash_table.DataTable(
            data=game_results,
            columns=[
                {"name": "Date", "id": "Date"},
                {"name": "Opponent", "id": "Opponent"},
                {"name": "Location", "id": "Location"},
                {"name": "Score", "id": "Score"},
                {"name": "Result", "id": "Result"},
            ],
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_cell={
                "textAlign": "left",
                "padding": "15px",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "rgb(248, 248, 248)",
                },
                {
                    "if": {"filter_query": "{Result} = 'W'"},
                    "backgroundColor": "rgba(0, 255, 0, 0.2)",
                },
                {
                    "if": {"filter_query": "{Result} = 'L'"},
                    "backgroundColor": "rgba(255, 0, 0, 0.2)",
                },
            ],
        )
