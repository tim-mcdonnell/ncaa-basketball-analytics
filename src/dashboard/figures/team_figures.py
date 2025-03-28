import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from typing import Optional, Dict, Any


def create_team_performance_chart(data: pl.DataFrame, team_name: Optional[str] = None) -> go.Figure:
    """
    Create a chart showing team performance over time.

    Args:
        data: DataFrame containing game results with the following columns:
            - game_date: Date of the game
            - points_scored: Points scored by the team
            - points_allowed: Points allowed by the team
            - win: 1 if the team won, 0 if the team lost
        team_name: Optional team name to display in the title

    Returns:
        A Plotly figure with the team performance chart
    """
    # Create figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Convert game_date to datetime if it's not already
    if isinstance(data["game_date"][0], str):
        data = data.with_columns(pl.col("game_date").str.to_date("%Y-%m-%d").alias("game_date"))

    # Sort by date
    data = data.sort("game_date")

    # Extract data for plotting
    dates = data["game_date"].to_list()
    points_scored = data["points_scored"].to_list()
    points_allowed = data["points_allowed"].to_list()
    win_status = data["win"].to_list()

    # Create traces for points
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=points_scored,
            name="Points Scored",
            line=dict(color="blue", width=2),
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=points_allowed,
            name="Points Allowed",
            line=dict(color="red", width=2),
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    # Add win/loss markers
    win_dates = [date for date, win in zip(dates, win_status) if win == 1]
    loss_dates = [date for date, win in zip(dates, win_status) if win == 0]

    win_points = [score for score, win in zip(points_scored, win_status) if win == 1]
    loss_points = [score for score, win in zip(points_scored, win_status) if win == 0]

    fig.add_trace(
        go.Scatter(
            x=win_dates,
            y=win_points,
            name="Win",
            mode="markers",
            marker=dict(color="green", size=12, symbol="circle"),
            showlegend=True,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=loss_dates,
            y=loss_points,
            name="Loss",
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            showlegend=True,
        ),
        secondary_y=False,
    )

    # Update layout
    title = f"{team_name} Performance" if team_name else "Team Performance"
    fig.update_layout(
        title=title,
        xaxis_title="Game Date",
        yaxis_title="Points",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig


def create_team_comparison_chart(
    team1_data: Dict[str, Any], team2_data: Dict[str, Any]
) -> go.Figure:
    """
    Create a radar chart comparing two teams.

    Args:
        team1_data: Dictionary with team 1 data:
            - name: Team name
            - stats: Dictionary of stat_name -> value
        team2_data: Dictionary with team 2 data:
            - name: Team name
            - stats: Dictionary of stat_name -> value

    Returns:
        A Plotly figure with the team comparison radar chart
    """
    # Get stat categories and values
    categories = list(team1_data["stats"].keys())
    team1_values = [team1_data["stats"][cat] for cat in categories]
    team2_values = [team2_data["stats"][cat] for cat in categories]

    # Add first value at the end to close the radar
    categories.append(categories[0])
    team1_values.append(team1_values[0])
    team2_values.append(team2_values[0])

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=team1_values,
            theta=categories,
            fill="toself",
            name=team1_data["name"],
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=team2_values,
            theta=categories,
            fill="toself",
            name=team2_data["name"],
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{team1_data['name']} vs. {team2_data['name']}",
        polar=dict(
            radialaxis=dict(
                visible=True,
            )
        ),
        showlegend=True,
        template="plotly_white",
    )

    return fig
