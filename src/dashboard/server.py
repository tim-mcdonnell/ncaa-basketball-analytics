from typing import Optional
import os

from .app import create_app


def get_server(debug: bool = False, host: Optional[str] = None, port: Optional[int] = None):
    """
    Create the Dash application and configure the server.

    Args:
        debug: Whether to run in debug mode
        host: Host to bind to
        port: Port to bind to

    Returns:
        Configured Dash application server
    """
    # Create the Dash app
    app = create_app()

    # Configure server settings
    host = host or os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = port or int(os.environ.get("DASHBOARD_PORT", "8050"))
    debug = debug or os.environ.get("DASHBOARD_DEBUG", "False").lower() == "true"

    # Register callbacks
    from .callbacks.team_callbacks import register_team_callbacks
    from .callbacks.game_callbacks import register_game_callbacks
    from .callbacks.player_callbacks import register_player_callbacks

    register_team_callbacks(app)
    register_game_callbacks(app)
    register_player_callbacks(app)

    return app.server, app, {"host": host, "port": port, "debug": debug}
