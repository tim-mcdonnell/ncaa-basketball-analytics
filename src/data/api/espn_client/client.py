"""ESPN API client implementation."""

from typing import Dict, Any, List, Optional
import logging
import os
import asyncio
import aiohttp

from src.data.api.async_client import AsyncClient
from src.data.api.rate_limiter import AdaptiveRateLimiter
from src.data.api.espn_client.config import ESPNConfig, load_espn_config
from src.data.api.exceptions import (
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    ConnectionResetError,
)
from src.data.api.espn_client.teams import TeamsEndpoint
from src.data.api.espn_client.games import GamesEndpoint
from src.data.api.espn_client.players import PlayersEndpoint

logger = logging.getLogger(__name__)


class AsyncESPNClient(AsyncClient, TeamsEndpoint, GamesEndpoint, PlayersEndpoint):
    """
    Asynchronous ESPN API client specifically for NCAA basketball data.

    This client extends the base AsyncClient with ESPN-specific endpoints
    and data processing for college basketball data.
    """

    def __init__(
        self,
        config: Optional[ESPNConfig] = None,
        config_path: Optional[str] = None,
        timeout: Optional[float] = None,
        metadata_dir: Optional[str] = None,
        metadata_file: Optional[str] = None,
        rate_limit_initial: Optional[int] = None,
        rate_limit_min: Optional[int] = None,
        rate_limit_max: Optional[int] = None,
        base_url: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_min_wait: Optional[float] = None,
        retry_max_wait: Optional[float] = None,
        retry_factor: Optional[float] = None,
    ):
        """
        Initialize ESPN API client.

        Args:
            config: Preconfigured ESPNConfig object, if provided
            config_path: Path to config file (default from config/api/espn.yaml)
            timeout: Optional timeout value to override config
            metadata_dir: Optional metadata directory to override config
            metadata_file: Optional metadata filename to override config
            rate_limit_initial: Optional initial rate limit to override config
            rate_limit_min: Optional minimum rate limit to override config
            rate_limit_max: Optional maximum rate limit to override config
            base_url: Optional base URL to override config
            max_retries: Optional max retries to override config
            retry_min_wait: Optional min wait time to override config
            retry_max_wait: Optional max wait time to override config
            retry_factor: Optional retry factor to override config
        """
        # Load config if not provided
        self.config = config or load_espn_config(config_path)

        # Override config with explicit parameters if provided
        effective_timeout = timeout or self.config.timeout
        effective_base_url = base_url or self.config.base_url
        effective_max_retries = max_retries or self.config.retries.max_attempts
        effective_retry_min_wait = retry_min_wait or self.config.retries.min_wait
        effective_retry_max_wait = retry_max_wait or self.config.retries.max_wait
        effective_retry_factor = retry_factor or self.config.retries.factor

        # Initialize base client
        super().__init__(
            base_url=effective_base_url,
            max_retries=effective_max_retries,
            retry_min_wait=effective_retry_min_wait,
            retry_max_wait=effective_retry_max_wait,
            retry_factor=effective_retry_factor,
            timeout=effective_timeout,
        )

        # Set up rate limiter with optional overrides
        self.rate_limiter = AdaptiveRateLimiter(
            initial=rate_limit_initial or self.config.rate_limiting.initial,
            min_limit=rate_limit_min or self.config.rate_limiting.min_limit,
            max_limit=rate_limit_max or self.config.rate_limiting.max_limit,
            success_threshold=self.config.rate_limiting.success_threshold,
            failure_threshold=self.config.rate_limiting.failure_threshold,
        )

        # Initialize metadata paths with optional overrides
        self.metadata_dir = metadata_dir or self.config.metadata.dir
        self.metadata_file = metadata_file or self.config.metadata.file

        logger.debug(
            f"Initialized AsyncESPNClient with metadata at {os.path.join(self.metadata_dir, self.metadata_file)}"
        )

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make GET request to ESPN API with rate limiting.

        Args:
            path: API endpoint path
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            APIError: If request fails
            RateLimitError: If rate limit exceeded
        """
        await self.rate_limiter.acquire()
        try:
            response = await super().get(path, params)
            await self.rate_limiter.release(success=True)
            return response
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                await self.rate_limiter.release(success=False)
                raise RateLimitError("ESPN API rate limit exceeded")
            await self.rate_limiter.release(success=False)
            raise APIError(f"ESPN API request failed: {str(e)}")
        except Exception as e:
            await self.rate_limiter.release(success=False)
            raise APIError(f"ESPN API request failed: {str(e)}")

    async def get_with_enhanced_recovery(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, max_recovery_attempts: int = 2
    ) -> Dict[str, Any]:
        """
        Make GET request with enhanced recovery for intermittent API issues.

        Args:
            endpoint: API endpoint
            params: Request parameters
            max_recovery_attempts: Maximum recovery attempts for intermittent issues

        Returns:
            Response data

        Raises:
            Various exceptions depending on the failure mode
        """
        recovery_attempt = 0
        last_exception = None

        # Regular retry mechanism will handle standard retries
        # This adds an additional recovery layer for intermittent issues
        while recovery_attempt <= max_recovery_attempts:
            try:
                return await self.get(endpoint, params)
            except (ServiceUnavailableError, ConnectionResetError) as e:
                last_exception = e
                recovery_attempt += 1

                # Use longer waits for recovery attempts
                backoff_time = min(30, 5 * 2**recovery_attempt)
                logger.warning(
                    f"Recovery attempt {recovery_attempt}/{max_recovery_attempts} "
                    f"for {endpoint}. Waiting {backoff_time}s before retry."
                )
                await asyncio.sleep(backoff_time)

        # If we exhausted recovery attempts, raise the last exception
        logger.error(f"Enhanced recovery failed after {max_recovery_attempts} attempts")
        raise last_exception


class ESPNClient:
    """
    Synchronous ESPN API client wrapper.

    This class provides a synchronous interface to the asynchronous
    AsyncESPNClient by using a thread-safe event loop to run
    asynchronous operations.
    """

    def __init__(
        self,
        config: Optional[ESPNConfig] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize synchronous ESPN client."""
        self._async_client = AsyncESPNClient(config=config, config_path=config_path)
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.close()

    # Team endpoints
    def get_teams(self, incremental: bool = False) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_teams."""
        return self._loop.run_until_complete(self._async_client.get_teams(incremental=incremental))

    def get_team_details(self, team_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for get_team_details."""
        return self._loop.run_until_complete(self._async_client.get_team_details(team_id))

    # Game endpoints
    def get_games(
        self,
        date_str: Optional[str] = None,
        team_id: Optional[str] = None,
        limit: int = 100,
        groups: Optional[str] = None,
        incremental: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_games."""
        return self._loop.run_until_complete(
            self._async_client.get_games(
                date_str=date_str,
                team_id=team_id,
                limit=limit,
                groups=groups,
                incremental=incremental,
                start_date=start_date,
                end_date=end_date,
            )
        )

    def get_game(self, game_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for get_game."""
        return self._loop.run_until_complete(self._async_client.get_game(game_id))

    # Player endpoints
    def get_team_players(self, team_id: str) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_team_players."""
        return self._loop.run_until_complete(self._async_client.get_team_players(team_id))

    def get_player_stats(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for get_player_stats."""
        return self._loop.run_until_complete(
            self._async_client.get_player_stats(player_id, season=season)
        )
