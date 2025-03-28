from dash import html, dcc
import dash_bootstrap_components as dbc


def create_layout():
    """
    Create the game prediction page layout.

    Returns:
        The game prediction page layout
    """
    layout = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("Game Prediction", className="mb-4"),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([html.Div(id="home-team-selector-container")], width=12, md=6),
                    dbc.Col([html.Div(id="away-team-selector-container")], width=12, md=6),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "Predict Game",
                                id="predict-game-button",
                                color="primary",
                                className="w-100 mb-4",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Game Prediction"),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="prediction-loading",
                                                type="circle",
                                                children=[
                                                    html.Div(
                                                        id="prediction-content",
                                                        className="text-center",
                                                    )
                                                ],
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Team Comparison"),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="team-comparison-loading",
                                                type="circle",
                                                children=[
                                                    dcc.Graph(
                                                        id="team-comparison-chart",
                                                        figure={},
                                                        config={"displayModeBar": False},
                                                    )
                                                ],
                                            )
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
        ]
    )

    return layout
