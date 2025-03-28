from dash import Dash, Input, Output, State, html
import plotly.graph_objects as go
import polars as pl
import random  # For demonstration purposes only

from ..components.selectors import TeamSelector
from ..data.repository import DashboardRepository
from ..figures.game_figures import create_game_prediction_chart, create_team_stat_comparison_chart


def register_game_callbacks(app: Dash) -> None:
    """
    Register callbacks for the game prediction page.

    Args:
        app: The Dash application instance
    """
    repository = DashboardRepository()

    # Home team selector callback
    @app.callback(
        Output("home-team-selector-container", "children"),
        Input("_", "children"),  # Dummy input for initialization
    )
    def populate_home_team_selector(_):
        """Populate the home team selector with teams from the database."""
        teams = repository.get_teams()
        team_options = [
            {"label": row["team_name"], "value": row["team_id"]} for row in teams.to_dicts()
        ]

        return html.Div(
            [
                html.H5("Home Team"),
                TeamSelector(
                    id="home-team-selector",
                    teams=team_options,
                    value=team_options[0]["value"] if team_options else None,
                ),
            ]
        )

    # Away team selector callback
    @app.callback(
        Output("away-team-selector-container", "children"),
        Input("_", "children"),  # Dummy input for initialization
    )
    def populate_away_team_selector(_):
        """Populate the away team selector with teams from the database."""
        teams = repository.get_teams()
        team_options = [
            {"label": row["team_name"], "value": row["team_id"]} for row in teams.to_dicts()
        ]

        return html.Div(
            [
                html.H5("Away Team"),
                TeamSelector(
                    id="away-team-selector",
                    teams=team_options,
                    value=team_options[1]["value"] if len(team_options) > 1 else None,
                ),
            ]
        )

    # Game prediction callback
    @app.callback(
        [
            Output("prediction-content", "children"),
            Output("team-comparison-chart", "figure"),
        ],
        Input("predict-game-button", "n_clicks"),
        [
            State("home-team-selector", "value"),
            State("away-team-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def predict_game(n_clicks, home_team_id, away_team_id):
        """Predict the outcome of a game between the selected teams."""
        if not home_team_id or not away_team_id:
            return html.Div("Please select both teams"), go.Figure()

        if home_team_id == away_team_id:
            return html.Div("Please select different teams for home and away"), go.Figure()

        # Get team names
        teams = repository.get_teams()
        home_team_df = teams.filter(pl.col("team_id") == home_team_id)
        away_team_df = teams.filter(pl.col("team_id") == away_team_id)

        home_team_name = home_team_df[0, "team_name"] if len(home_team_df) > 0 else "Home Team"
        away_team_name = away_team_df[0, "team_name"] if len(away_team_df) > 0 else "Away Team"

        # In a real application, this would be a machine learning prediction
        # For demonstration, we'll use a random prediction
        home_win_probability = random.uniform(0.3, 0.7)

        # Create prediction content
        if home_win_probability > 0.5:
            prediction_result = f"{home_team_name} is favored to win"
            win_color = "green"
        else:
            prediction_result = f"{away_team_name} is favored to win"
            win_color = "red"

        prediction_content = html.Div(
            [
                create_game_prediction_chart(home_team_name, away_team_name, home_win_probability),
                html.H4(prediction_result, style={"color": win_color, "marginTop": "20px"}),
                html.P(f"Home Team Win Probability: {home_win_probability:.1%}"),
                html.P(f"Away Team Win Probability: {1-home_win_probability:.1%}"),
            ]
        )

        # Create team comparison chart
        # In a real application, these stats would come from the database
        # For demonstration, we'll create sample statistics
        home_stats = {
            "Points Per Game": 75.2 + random.uniform(-5, 5),
            "Field Goal %": 45.3 + random.uniform(-3, 3),
            "3-Point %": 36.1 + random.uniform(-3, 3),
            "Free Throw %": 72.5 + random.uniform(-3, 3),
            "Rebounds": 35.8 + random.uniform(-4, 4),
            "Assists": 15.2 + random.uniform(-3, 3),
            "Steals": 7.1 + random.uniform(-2, 2),
            "Blocks": 3.5 + random.uniform(-1, 1),
            "Turnovers": 12.3 + random.uniform(-2, 2),
        }

        away_stats = {
            "Points Per Game": 72.8 + random.uniform(-5, 5),
            "Field Goal %": 44.1 + random.uniform(-3, 3),
            "3-Point %": 35.2 + random.uniform(-3, 3),
            "Free Throw %": 71.9 + random.uniform(-3, 3),
            "Rebounds": 34.2 + random.uniform(-4, 4),
            "Assists": 14.5 + random.uniform(-3, 3),
            "Steals": 6.8 + random.uniform(-2, 2),
            "Blocks": 3.2 + random.uniform(-1, 1),
            "Turnovers": 11.9 + random.uniform(-2, 2),
        }

        comparison_chart = create_team_stat_comparison_chart(
            home_stats, away_stats, home_team_name, away_team_name
        )

        return prediction_content, comparison_chart
