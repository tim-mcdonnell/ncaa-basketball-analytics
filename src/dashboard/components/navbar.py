from dash import html
import dash_bootstrap_components as dbc
from typing import List, Dict, Optional


class Navbar(html.Div):
    """A component for the dashboard navigation bar."""

    def __init__(
        self,
        pages: Optional[List[Dict[str, str]]] = None,
        logo_url: Optional[str] = None,
        brand_name: str = "NCAA Basketball Analytics",
        className: str = "",
    ):
        """
        Initialize the navbar component.

        Args:
            pages: List of page dictionaries with 'name' and 'path' keys
            logo_url: Optional URL for a logo image
            brand_name: The name to display in the navbar
            className: Additional CSS classes
        """
        # Default pages if none provided
        if pages is None:
            pages = [
                {"name": "Team Analysis", "path": "/team-analysis"},
                {"name": "Game Prediction", "path": "/game-prediction"},
                {"name": "Player Statistics", "path": "/player-statistics"},
            ]

        # Create navigation links
        nav_links = []
        for page in pages:
            nav_links.append(
                dbc.NavItem(
                    dbc.NavLink(
                        page["name"],
                        href=page["path"],
                        active="exact",
                    )
                )
            )

        # Create brand content with optional logo
        brand_content = []
        if logo_url:
            brand_content.append(
                html.Img(
                    src=logo_url,
                    height="30px",
                    className="me-2",
                )
            )
        brand_content.append(brand_name)

        # Create navbar
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        brand_content,
                        href="/",
                        className="navbar-brand",
                    ),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Nav(
                            nav_links,
                            className="ms-auto",
                            navbar=True,
                        ),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ]
            ),
            color="primary",
            dark=True,
            className="mb-4",
        )

        super().__init__(
            children=navbar,
            className=f"navbar-component {className}",
        )
