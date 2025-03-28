from dash import html, dcc
import dash_bootstrap_components as dbc


def create_layout():
    """
    Create the team analysis page layout.

    Returns:
        The team analysis page layout
    """
    layout = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("Team Analysis", className="mb-4"),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([html.Div(id="team-selector-container")], width=12, lg=3),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Team Performance"),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="team-performance-loading",
                                                type="circle",
                                                children=[
                                                    dcc.Graph(
                                                        id="team-performance-chart",
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
                        lg=9,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Team Statistics"),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="team-stats-loading",
                                                type="circle",
                                                children=[html.Div(id="team-stats-table")],
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Recent Games"),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                id="recent-games-loading",
                                                type="circle",
                                                children=[html.Div(id="recent-games-table")],
                                            )
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
        ]
    )

    return layout
