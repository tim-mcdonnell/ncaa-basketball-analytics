from dash import Dash, Input, Output, html, dcc, dash_table
import plotly.graph_objects as go
import polars as pl

from ..components.selectors import TeamSelector, MetricSelector
from ..data.repository import DashboardRepository
from ..figures.player_figures import create_player_performance_chart, create_player_comparison_chart


def register_player_callbacks(app: Dash) -> None:
    """
    Register callbacks for the player statistics page.

    Args:
        app: The Dash application instance
    """
    repository = DashboardRepository()

    # Team selector callback
    @app.callback(
        Output("team-selector-player-container", "children"),
        Input("_", "children"),  # Dummy input for initialization
    )
    def populate_team_selector(_):
        """Populate the team selector with teams from the database."""
        teams = repository.get_teams()
        team_options = [
            {"label": row["team_name"], "value": row["team_id"]} for row in teams.to_dicts()
        ]

        return TeamSelector(
            id="team-selector-player",
            teams=team_options,
            value=team_options[0]["value"] if team_options else None,
        )

    # Player selector callback
    @app.callback(
        Output("player-selector-container", "children"),
        Input("team-selector-player", "value"),
    )
    def populate_player_selector(team_id):
        """Populate the player selector with players from the selected team."""
        if not team_id:
            return html.Div("Select a team first")

        players = repository.get_players(team_id)
        player_options = [
            {"label": row["player_name"], "value": row["player_id"]} for row in players.to_dicts()
        ]

        return TeamSelector(  # Reusing TeamSelector for players too
            id="player-selector",
            label="Select Player",
            placeholder="Select a player...",
            teams=player_options,
            value=player_options[0]["value"] if player_options else None,
        )

    # Metric selector callback
    @app.callback(
        Output("metric-selector-container", "children"),
        Input("_", "children"),  # Dummy input for initialization
    )
    def populate_metric_selector(_):
        """Populate the metric selector with available metrics."""
        metrics = [
            {"label": "Points", "value": "points"},
            {"label": "Rebounds", "value": "rebounds"},
            {"label": "Assists", "value": "assists"},
            {"label": "Steals", "value": "steals"},
            {"label": "Blocks", "value": "blocks"},
        ]

        return MetricSelector(
            id="metric-selector",
            metrics=metrics,
            value=["points"],
        )

    # Player performance chart callback
    @app.callback(
        Output("player-performance-chart", "figure"),
        [
            Input("player-selector", "value"),
            Input("metric-selector", "value"),
        ],
    )
    def update_player_performance_chart(player_id, metrics):
        """Update the player performance chart based on the selected player and metrics."""
        if not player_id or not metrics:
            return go.Figure()

        # Get player name
        # In a real app, we would get this from the database
        # For now, we'll use a placeholder
        player_name = f"Player {player_id}"

        # Get player stats
        stats = repository.get_player_stats(player_id)

        # Use the first selected metric for the performance chart
        metric = metrics[0]

        # Create the chart
        return create_player_performance_chart(stats, metric, player_name)

    # Player stats table callback
    @app.callback(
        Output("player-stats-table", "children"),
        Input("player-selector", "value"),
    )
    def update_player_stats_table(player_id):
        """Update the player statistics table based on the selected player."""
        if not player_id:
            return html.Div("Select a player to view statistics")

        # Get player stats
        stats = repository.get_player_stats(player_id)

        if len(stats) == 0:
            return html.Div("No statistics found for the selected player")

        # Calculate averages
        avg_points = stats["points"].mean()
        avg_rebounds = stats["rebounds"].mean()
        avg_assists = stats["assists"].mean()
        avg_steals = stats["steals"].mean()
        avg_blocks = stats["blocks"].mean()

        # Create statistics data
        stats_data = [
            {"Statistic": "Games Played", "Value": len(stats)},
            {"Statistic": "Points Per Game", "Value": f"{avg_points:.1f}"},
            {"Statistic": "Rebounds Per Game", "Value": f"{avg_rebounds:.1f}"},
            {"Statistic": "Assists Per Game", "Value": f"{avg_assists:.1f}"},
            {"Statistic": "Steals Per Game", "Value": f"{avg_steals:.1f}"},
            {"Statistic": "Blocks Per Game", "Value": f"{avg_blocks:.1f}"},
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

    # Player comparison callback
    @app.callback(
        Output("player-comparison-container", "children"),
        [
            Input("team-selector-player", "value"),
            Input("metric-selector", "value"),
        ],
    )
    def update_player_comparison(team_id, metrics):
        """Update the player comparison container with a chart comparing players."""
        if not team_id or not metrics:
            return html.Div("Select a team and metrics for comparison")

        # Get players for the team
        players = repository.get_players(team_id)

        if len(players) == 0:
            return html.Div("No players found for the selected team")

        # Get stats for each player and calculate averages
        comparison_data = []
        for player in players.to_dicts():
            player_id = player["player_id"]
            player_stats = repository.get_player_stats(player_id)

            if len(player_stats) > 0:
                player_row = {"player_name": player["player_name"]}

                # Calculate averages for each metric
                for metric in metrics:
                    player_row[metric] = player_stats[metric].mean()

                comparison_data.append(player_row)

        if not comparison_data:
            return html.Div("No statistics available for comparison")

        # Create comparison DataFrame
        comparison_df = pl.DataFrame(comparison_data)

        # Create comparison chart
        comparison_chart = create_player_comparison_chart(comparison_df, metrics)

        return dcc.Graph(
            figure=comparison_chart,
            config={"displayModeBar": False},
        )
