"""ESPN API client for NCAA Basketball data."""

from src.data.api.espn_client.client import ESPNClient, AsyncESPNClient
from src.data.api.espn_client.config import ESPNConfig, load_espn_config
from src.data.api.espn_client.teams import TeamsEndpoint
from src.data.api.espn_client.games import GamesEndpoint
from src.data.api.espn_client.players import PlayersEndpoint
from src.data.api.espn_client.adapter import ESPNApiClient

__all__ = [
    "ESPNClient",
    "AsyncESPNClient",
    "ESPNConfig",
    "load_espn_config",
    "TeamsEndpoint",
    "GamesEndpoint",
    "PlayersEndpoint",
    "ESPNApiClient",
]
