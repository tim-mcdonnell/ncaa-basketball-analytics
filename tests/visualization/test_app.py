from dash import Dash


def test_app_initialization():
    """Verify the Dash app initializes correctly."""
    # Import app creation function
    from src.dashboard.app import create_app

    app = create_app()

    # Assert app is created correctly
    assert isinstance(app, Dash)
    assert app.title == "NCAA Basketball Analytics"

    # Assert layout is created
    assert app.layout is not None

    # Verify that the app has a tabs component
    # Convert the layout tree to a string and check if it contains the tabs ID
    layout_str = str(app.layout)
    assert "tabs" in layout_str and "Team Analysis" in layout_str
    assert "Game Prediction" in layout_str
    assert "Player Statistics" in layout_str
