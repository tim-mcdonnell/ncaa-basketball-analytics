import plotly.graph_objects as go
from typing import Dict


def create_game_prediction_chart(
    home_team: str, away_team: str, home_win_probability: float
) -> go.Figure:
    """
    Create a chart showing game prediction.

    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        home_win_probability: Probability of the home team winning (0-1)

    Returns:
        A Plotly figure with the game prediction chart
    """
    # Create figure with gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=home_win_probability * 100,
            title={"text": f"{home_team} Win Probability"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "blue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 100], "color": "lightblue"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            number={"suffix": "%", "valueformat": ".1f"},
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{home_team} vs. {away_team} Prediction",
        margin=dict(l=20, r=20, t=60, b=20),
        height=300,
    )

    return fig


def create_team_stat_comparison_chart(
    home_stats: Dict[str, float], away_stats: Dict[str, float], home_team: str, away_team: str
) -> go.Figure:
    """
    Create a horizontal bar chart comparing team statistics.

    Args:
        home_stats: Dictionary of home team statistics (stat_name -> value)
        away_stats: Dictionary of away team statistics (stat_name -> value)
        home_team: Name of the home team
        away_team: Name of the away team

    Returns:
        A Plotly figure with the team statistic comparison chart
    """
    # Get common stats
    stats = sorted(set(home_stats.keys()).intersection(set(away_stats.keys())))

    # Create a figure with a horizontal bar chart
    fig = go.Figure()

    # Add home team stats
    home_values = [home_stats[stat] for stat in stats]
    fig.add_trace(
        go.Bar(
            y=stats,
            x=home_values,
            name=home_team,
            orientation="h",
            marker=dict(color="blue"),
        )
    )

    # Add away team stats
    away_values = [away_stats[stat] for stat in stats]
    fig.add_trace(
        go.Bar(
            y=stats,
            x=away_values,
            name=away_team,
            orientation="h",
            marker=dict(color="red"),
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{home_team} vs. {away_team} Statistical Comparison",
        xaxis_title="Value",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
