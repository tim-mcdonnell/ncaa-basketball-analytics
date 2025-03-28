from dash import html
from typing import Optional


class Header(html.Div):
    """A component for the dashboard header."""

    def __init__(
        self,
        title: str = "NCAA Basketball Analytics",
        subtitle: Optional[str] = None,
        logo_url: Optional[str] = None,
        className: str = "",
    ):
        """
        Initialize the header component.

        Args:
            title: The main title to display
            subtitle: Optional subtitle to display
            logo_url: Optional URL for a logo image
            className: Additional CSS classes
        """
        header_elements = []

        # Add logo if provided
        if logo_url:
            header_elements.append(
                html.Img(
                    src=logo_url,
                    height="40px",
                    className="me-2",
                )
            )

        # Add title
        header_elements.append(
            html.Div(
                children=[
                    html.H1(
                        title,
                        className="mb-0 text-primary",
                    ),
                    html.P(
                        subtitle or "",
                        className="text-muted mb-0" if subtitle else "d-none",
                    ),
                ],
            )
        )

        super().__init__(
            children=header_elements,
            className=f"d-flex align-items-center py-3 {className}",
        )
