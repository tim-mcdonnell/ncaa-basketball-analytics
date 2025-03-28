from dash import Dash, html
import dash_bootstrap_components as dbc

# Import layouts
from .layouts.team_analysis import create_layout as create_team_layout
from .layouts.game_prediction import create_layout as create_game_layout
from .layouts.player_statistics import create_layout as create_player_layout


def create_app(title: str = "NCAA Basketball Analytics") -> Dash:
    """
    Create and configure the Dash application instance.

    Args:
        title: The title of the application

    Returns:
        Configured Dash application
    """
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    app.title = title

    # Define routes and layouts
    app.layout = html.Div(
        [
            dbc.Container(
                dbc.Tabs(
                    [
                        dbc.Tab(
                            create_team_layout(), label="Team Analysis", tab_id="team-analysis"
                        ),
                        dbc.Tab(
                            create_game_layout(), label="Game Prediction", tab_id="game-prediction"
                        ),
                        dbc.Tab(
                            create_player_layout(),
                            label="Player Statistics",
                            tab_id="player-statistics",
                        ),
                    ],
                    id="tabs",
                    active_tab="team-analysis",
                ),
                fluid=True,
                className="px-4 py-3",
            )
        ]
    )

    return app
