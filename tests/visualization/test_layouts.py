from dash import html


def test_layout_rendering():
    """Verify each page layout renders without errors."""
    # Import layouts directly
    from src.dashboard.layouts.team_analysis import create_layout as create_team_layout
    from src.dashboard.layouts.game_prediction import create_layout as create_game_layout
    from src.dashboard.layouts.player_statistics import create_layout as create_player_layout

    # Test team analysis layout
    team_layout = create_team_layout()
    assert team_layout is not None
    assert isinstance(team_layout, html.Div)

    # Test game prediction layout
    game_layout = create_game_layout()
    assert game_layout is not None
    assert isinstance(game_layout, html.Div)

    # Test player statistics layout
    player_layout = create_player_layout()
    assert player_layout is not None
    assert isinstance(player_layout, html.Div)
