import pytest
from unittest.mock import MagicMock, AsyncMock

from src.data.api.async_client import AsyncClient
from src.data.api.espn_client import AsyncESPNClient
from src.data.api.rate_limiter import AdaptiveRateLimiter


@pytest.fixture
def mock_async_client():
    """Create a mock AsyncClient that doesn't make real requests."""
    client = MagicMock(spec=AsyncClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock()
    client.post = AsyncMock()
    return client


@pytest.fixture
def mock_espn_client():
    """Create a mock AsyncESPNClient that doesn't make real requests."""
    client = MagicMock(spec=AsyncESPNClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock()
    client.get_teams = AsyncMock()
    client.get_team_details = AsyncMock()
    client.get_games = AsyncMock()
    client.get_team_players = AsyncMock()

    # Set up rate limiter mock
    client.rate_limiter = MagicMock(spec=AdaptiveRateLimiter)
    client.rate_limiter.acquire = AsyncMock()
    client.rate_limiter.release = AsyncMock()

    return client


@pytest.fixture
def sample_teams_response():
    """Sample response for teams endpoint."""
    return {
        "sports": [
            {
                "leagues": [
                    {
                        "teams": [
                            {"team": {"id": "1", "name": "Team A", "abbreviation": "TA"}},
                            {"team": {"id": "2", "name": "Team B", "abbreviation": "TB"}},
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_team_details_response():
    """Sample response for team details endpoint."""
    return {
        "team": {
            "id": "1",
            "name": "Team A",
            "abbreviation": "TA",
            "location": "Location A",
            "logo": "http://example.com/logo.png",
            "record": {"items": [{"summary": "10-5"}]},
        }
    }


@pytest.fixture
def sample_games_response():
    """Sample response for games endpoint."""
    return {
        "events": [
            {
                "id": "401",
                "date": "2023-11-15T19:00Z",
                "name": "Team A vs Team B",
                "shortName": "TA vs TB",
                "competitions": [
                    {
                        "competitors": [
                            {"team": {"id": "1", "name": "Team A"}, "score": "75"},
                            {"team": {"id": "2", "name": "Team B"}, "score": "70"},
                        ],
                        "status": {"type": {"completed": True}},
                    }
                ],
            },
            {
                "id": "402",
                "date": "2023-11-18T20:00Z",
                "name": "Team C vs Team D",
                "shortName": "TC vs TD",
                "competitions": [
                    {
                        "competitors": [
                            {"team": {"id": "3", "name": "Team C"}, "score": "80"},
                            {"team": {"id": "4", "name": "Team D"}, "score": "82"},
                        ],
                        "status": {"type": {"completed": True}},
                    }
                ],
            },
        ]
    }


@pytest.fixture
def sample_players_response():
    """Sample response for team players endpoint."""
    return {
        "athletes": [
            {"id": "101", "fullName": "Player One", "position": {"name": "Guard"}, "jersey": "10"},
            {
                "id": "102",
                "fullName": "Player Two",
                "position": {"name": "Forward"},
                "jersey": "20",
            },
        ]
    }
