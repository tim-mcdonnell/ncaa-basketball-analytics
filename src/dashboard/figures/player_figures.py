import plotly.graph_objects as go
import polars as pl
from typing import Optional, List


def create_player_performance_chart(
    data: pl.DataFrame, metric: str, player_name: Optional[str] = None
) -> go.Figure:
    """
    Create a chart showing player performance over time.

    Args:
        data: DataFrame containing player stats with the following columns:
            - game_date: Date of the game
            - {metric}: The metric to chart (points, rebounds, etc.)
        metric: The metric to chart (points, rebounds, etc.)
        player_name: Optional player name to display in the title

    Returns:
        A Plotly figure with the player performance chart
    """
    # Convert game_date to datetime if it's not already
    if isinstance(data["game_date"][0], str):
        data = data.with_columns(pl.col("game_date").str.to_date("%Y-%m-%d").alias("game_date"))

    # Sort by date
    data = data.sort("game_date")

    # Create figure
    fig = go.Figure()

    # Add line trace for the metric
    fig.add_trace(
        go.Scatter(
            x=data["game_date"].to_list(),
            y=data[metric].to_list(),
            mode="lines+markers",
            name=metric.replace("_", " ").title(),
            line=dict(color="blue", width=2),
            marker=dict(size=8),
        )
    )

    # Add moving average if enough data points
    if len(data) >= 3:
        window_size = min(3, len(data))
        moving_avg = data[metric].rolling_mean(window_size)

        fig.add_trace(
            go.Scatter(
                x=data["game_date"].to_list(),
                y=moving_avg.to_list(),
                mode="lines",
                name=f"{window_size}-Game Average",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

    # Update layout
    metric_name = metric.replace("_", " ").title()
    title = f"{player_name} {metric_name}" if player_name else f"Player {metric_name}"

    fig.update_layout(
        title=title,
        xaxis_title="Game Date",
        yaxis_title=metric_name,
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_player_comparison_chart(data: pl.DataFrame, metrics: List[str]) -> go.Figure:
    """
    Create a bar chart comparing players.

    Args:
        data: DataFrame containing player data with columns:
            - player_name: Name of the player
            - {metric} for each metric in metrics: Values for each metric
        metrics: List of metrics to compare

    Returns:
        A Plotly figure with the player comparison chart
    """
    # Get player names
    player_names = data["player_name"].to_list()

    # Convert metric names for display
    metric_display_names = [m.replace("_", " ").title() for m in metrics]

    # Create a grouped bar chart
    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=player_names,
                y=data[metric].to_list(),
                name=metric_display_names[i],
            )
        )

    # Update layout
    fig.update_layout(
        title="Player Comparison",
        xaxis_title="Player",
        yaxis_title="Value",
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
