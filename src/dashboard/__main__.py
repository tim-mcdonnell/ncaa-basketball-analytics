from .server import get_server


def main():
    """
    Run the dashboard server.
    """
    # Get the server configuration
    server, app, config = get_server(debug=True)

    # Run the app
    app.run(host=config["host"], port=config["port"], debug=config["debug"])


if __name__ == "__main__":
    main()
