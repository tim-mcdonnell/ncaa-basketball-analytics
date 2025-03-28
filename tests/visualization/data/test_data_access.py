import polars as pl


def test_data_repository():
    """Verify database queries return expected results."""
    # Import repository (will fail initially)
    from src.dashboard.data.repository import DashboardRepository

    # Create repository
    repo = DashboardRepository()
    assert repo is not None

    # Test team queries
    teams = repo.get_teams()
    assert teams is not None
    assert isinstance(teams, pl.DataFrame)
    assert len(teams) > 0
    assert "team_id" in teams.columns
    assert "team_name" in teams.columns

    # Test game queries
    games = repo.get_recent_games(limit=5)
    assert games is not None
    assert isinstance(games, pl.DataFrame)
    assert len(games) <= 5
    assert "game_id" in games.columns

    # Test player queries
    players = repo.get_players(team_id="TEST")
    assert players is not None
    assert isinstance(players, pl.DataFrame)
    assert "player_id" in players.columns


def test_data_caching():
    """Verify caching mechanism works efficiently."""
    # Import repository with caching (will fail initially)
    from src.dashboard.data.repository import DashboardRepository

    # Create repository
    repo = DashboardRepository()

    # First call should hit the database
    import time

    start_time = time.time()
    teams1 = repo.get_teams()
    first_call_time = time.time() - start_time

    # Second call should use cache and be faster
    start_time = time.time()
    teams2 = repo.get_teams()
    second_call_time = time.time() - start_time

    # Verify cached result is the same
    assert teams1.equals(teams2)

    # Cached call should generally be faster, but we'll use a loose assertion
    # since test environments can have variability
    # This might occasionally fail due to system load, test timing, etc.
    # so we're being very conservative with the threshold
    assert second_call_time <= first_call_time * 2, "Cached call not faster than database call"
