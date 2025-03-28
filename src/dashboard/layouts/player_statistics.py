from dash import html, dcc
import dash_bootstrap_components as dbc


def create_layout():
    """
    Create the player statistics page layout.

    Returns:
        The player statistics page layout
    """
    layout = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("Player Statistics", className="mb-4"),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([html.Div(id="team-selector-player-container")], width=12, lg=4),
                    dbc.Col([html.Div(id="player-selector-container")], width=12, lg=4),
                    dbc.Col([html.Div(id="metric-selector-container")], width=12, lg=4),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Player Performance"),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="player-performance-loading",
                                                type="circle",
                                                children=[
                                                    dcc.Graph(
                                                        id="player-performance-chart",
                                                        figure={},
                                                        config={"displayModeBar": False},
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
                                    dbc.CardHeader("Player Statistics"),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="player-stats-loading",
                                                type="circle",
                                                children=[html.Div(id="player-stats-table")],
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
                                    dbc.CardHeader("Player Comparison"),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="player-comparison-loading",
                                                type="circle",
                                                children=[
                                                    html.Div(id="player-comparison-container")
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
