"""Dashboard configuration model."""

from typing import List

from pydantic import Field, field_validator

from src.config.base import BaseConfig


class DashboardTheme(BaseConfig):
    """Dashboard theme configuration.

    This model defines theming options for the dashboard.

    Example YAML configuration:
    ```yaml
    dashboard:
      theme:
        primary_color: "#0066cc"
        secondary_color: "#f8f9fa"
        text_color: "#212529"
        background_color: "#ffffff"
        font_family: "Arial, sans-serif"
    ```
    """

    primary_color: str = Field(default="#0066cc", description="Primary color for the dashboard")
    secondary_color: str = Field(default="#f8f9fa", description="Secondary color for the dashboard")
    text_color: str = Field(default="#212529", description="Text color for the dashboard")
    background_color: str = Field(
        default="#ffffff", description="Background color for the dashboard"
    )
    font_family: str = Field(
        default="Arial, sans-serif", description="Font family for the dashboard"
    )

    @field_validator("primary_color", "secondary_color", "text_color", "background_color")
    @classmethod
    def validate_color(cls, v: str) -> str:
        """Validate that the color is a valid hex code."""
        if not v.startswith("#") or not all(c in "0123456789ABCDEFabcdef" for c in v[1:]):
            raise ValueError(f"Invalid color code: {v}. Must be a valid hex color code.")
        return v


class DashboardLayout(BaseConfig):
    """Dashboard layout configuration.

    This model defines layout options for the dashboard.

    Example YAML configuration:
    ```yaml
    dashboard:
      layout:
        sidebar_width: 250
        content_width: "fluid"
        charts_per_row: 2
        default_chart_height: 400
    ```
    """

    sidebar_width: int = Field(default=250, description="Width of the sidebar in pixels", ge=0)
    content_width: str = Field(
        default="fluid", description="Width of the content area ('fixed' or 'fluid')"
    )
    charts_per_row: int = Field(
        default=2, description="Number of charts to display per row", ge=1, le=4
    )
    default_chart_height: int = Field(
        default=400, description="Default height for charts in pixels", ge=100
    )

    @field_validator("content_width")
    @classmethod
    def validate_content_width(cls, v: str) -> str:
        """Validate that the content width is either 'fixed' or 'fluid'."""
        if v not in ["fixed", "fluid"]:
            raise ValueError(f"Invalid content width: {v}. Must be either 'fixed' or 'fluid'.")
        return v


class DashboardConfig(BaseConfig):
    """Dashboard configuration model.

    This model defines configuration for the dashboard component.

    Example YAML configuration:
    ```yaml
    dashboard:
      title: "NCAA Basketball Analytics"
      refresh_interval: 60
      theme:
        primary_color: "#0066cc"
        secondary_color: "#f8f9fa"
      layout:
        sidebar_width: 250
        content_width: "fluid"
      default_view: "team_comparison"
      available_views:
        - "team_overview"
        - "team_comparison"
        - "player_stats"
        - "predictions"
      cache_timeout: 300
    ```
    """

    title: str = Field(default="NCAA Basketball Analytics", description="Title of the dashboard")
    refresh_interval: int = Field(
        default=60,
        description="Dashboard refresh interval in seconds (0 for manual refresh only)",
        ge=0,
    )
    theme: DashboardTheme = Field(
        default_factory=DashboardTheme, description="Dashboard theme configuration"
    )
    layout: DashboardLayout = Field(
        default_factory=DashboardLayout, description="Dashboard layout configuration"
    )
    default_view: str = Field(
        default="team_overview", description="Default view to display when the dashboard loads"
    )
    available_views: List[str] = Field(
        default_factory=lambda: ["team_overview", "team_comparison", "player_stats", "predictions"],
        description="List of available views in the dashboard",
    )
    cache_timeout: int = Field(
        default=300, description="Cache timeout for dashboard data in seconds", ge=0
    )
    show_filters: bool = Field(
        default=True, description="Whether to show filter controls in the dashboard"
    )
    max_items_per_page: int = Field(
        default=25,
        description="Maximum number of items to display per page in tables",
        ge=5,
        le=100,
    )

    @field_validator("default_view")
    @classmethod
    def validate_default_view(cls, v: str, values) -> str:
        """Validate that the default view is in the list of available views."""
        available_views = values.data.get("available_views")
        if available_views and v not in available_views:
            raise ValueError(
                f"Default view '{v}' must be one of the available views: {available_views}"
            )
        return v
