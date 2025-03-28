import plotly.graph_objects as go
import polars as pl


def test_team_performance_chart():
    """Verify team performance chart generation works correctly."""
    # Import figure generator (will fail initially)
    from src.dashboard.figures.team_figures import create_team_performance_chart

    # Create test data
    data = pl.DataFrame(
        {
            "game_date": ["2023-01-01", "2023-01-05", "2023-01-10"],
            "points_scored": [75, 82, 68],
            "points_allowed": [70, 75, 72],
            "win": [1, 1, 0],
        }
    )

    # Create figure
    fig = create_team_performance_chart(data)

    # Verify figure is created correctly
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Should have at least one trace
    assert fig.layout.title is not None


def test_player_comparison_chart():
    """Verify player comparison chart generation works correctly."""
    # Import figure generator (will fail initially)
    from src.dashboard.figures.player_figures import create_player_comparison_chart

    # Create test data
    data = pl.DataFrame(
        {
            "player_name": ["Player A", "Player B"],
            "points_per_game": [15.5, 12.3],
            "rebounds_per_game": [5.2, 8.7],
            "assists_per_game": [4.3, 2.1],
        }
    )

    # Create figure
    fig = create_player_comparison_chart(
        data, metrics=["points_per_game", "rebounds_per_game", "assists_per_game"]
    )

    # Verify figure is created correctly
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Should have at least one trace
    assert fig.layout.title is not None


def test_game_prediction_chart():
    """Verify game prediction chart generation works correctly."""
    # Import figure generator (will fail initially)
    from src.dashboard.figures.game_figures import create_game_prediction_chart

    # Create test data
    home_team = "Team A"
    away_team = "Team B"
    home_win_probability = 0.65

    # Create figure
    fig = create_game_prediction_chart(home_team, away_team, home_win_probability)

    # Verify figure is created correctly
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Should have at least one trace
    assert fig.layout.title is not None
    assert home_team in str(fig.layout.title.text)
    assert away_team in str(fig.layout.title.text)
