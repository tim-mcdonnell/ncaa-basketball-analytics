import aiohttp
from typing import Dict, Any, Optional, Callable, TypeVar, Type, Awaitable
import logging
import asyncio
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_not_exception_type,
    before_sleep_log,
)

from src.data.api.exceptions import (
    APIError,
    RateLimitError,
    AuthenticationError,
    ResourceNotFoundError,
    ValidationError,
    ServiceUnavailableError,
    TimeoutError as APITimeoutError,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class AsyncClient:
    """
    Base asynchronous HTTP client for API requests.
    Provides common functionality for making HTTP requests using aiohttp.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_min_wait: float = 1,
        retry_max_wait: float = 30,
        retry_factor: float = 2,
    ):
        """
        Initialize the asynchronous client.

        Args:
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            headers: Optional default headers for requests
            max_retries: Maximum number of retry attempts
            retry_min_wait: Minimum wait time between retries in seconds
            retry_max_wait: Maximum wait time between retries in seconds
            retry_factor: Exponential backoff factor
        """
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers or {}
        self.session = None
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.retry_factor = retry_factor
        # Create the retry decorator during initialization
        self._retry_decorator = self._create_retry_decorator(
            retry_on_exceptions=(aiohttp.ClientError, asyncio.TimeoutError, ServiceUnavailableError)
        )

    async def __aenter__(self):
        """Set up client session when entering async context."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout), headers=self.headers
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close client session when exiting async context."""
        if self.session and not self.session.closed:
            await self.session.close()

    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL from endpoint.

        Args:
            endpoint: API endpoint path

        Returns:
            Full URL with base URL and endpoint
        """
        # Ensure endpoint has leading slash and handle any trailing slashes in base_url
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        if self.base_url.endswith("/"):
            return f"{self.base_url[:-1]}{endpoint}"
        return f"{self.base_url}{endpoint}"

    def _create_retry_decorator(
        self,
        retry_on_exceptions: Optional[Type[Exception]] = None,
        skip_on_exceptions: Optional[Type[Exception]] = None,
    ) -> Callable:
        """
        Create a tenacity retry decorator with the client's retry settings.

        Args:
            retry_on_exceptions: Exception types to retry on
            skip_on_exceptions: Exception types to not retry on

        Returns:
            Retry decorator
        """
        retry_conditions = []
        if retry_on_exceptions:
            retry_conditions.append(retry_if_exception_type(retry_on_exceptions))
        if skip_on_exceptions:
            retry_conditions.append(retry_if_not_exception_type(skip_on_exceptions))

        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(
                multiplier=self.retry_factor, min=self.retry_min_wait, max=self.retry_max_wait
            ),
            retry=retry_conditions[0] if len(retry_conditions) == 1 else retry_conditions,
            before_sleep=before_sleep_log(logger, logging.INFO),
        )

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Handle API response and convert to appropriate error if needed.

        Args:
            response: The HTTP response

        Returns:
            Parsed JSON response

        Raises:
            Various APIError subclasses based on status code
        """
        status_code = response.status

        try:
            data = await response.json()
        except (aiohttp.ContentTypeError, aiohttp.ClientError):
            # Handle non-JSON responses
            text = await response.text()
            if status_code >= 400:
                logger.error(f"API error {status_code}: {text}")
                self._raise_error_by_status(status_code, text)
            return {"text": text}  # Return text as data for successful non-JSON responses

        if status_code >= 400:
            logger.error(f"API error {status_code}: {data}")
            self._raise_error_by_status(status_code, data)

        return data

    def _raise_error_by_status(self, status_code: int, data: Any) -> None:
        """
        Raise appropriate error based on status code.

        Args:
            status_code: HTTP status code
            data: Response data

        Raises:
            Various APIError subclasses
        """
        error_msg = str(data) if not isinstance(data, dict) else data.get("error", str(data))

        if status_code == 400:
            errors = data.get("errors") if isinstance(data, dict) else None
            raise ValidationError(error_msg, errors)
        elif status_code == 401:
            raise AuthenticationError(error_msg)
        elif status_code == 404:
            resource_type = "Resource"
            resource_id = "unknown"
            raise ResourceNotFoundError(resource_type, resource_id)
        elif status_code == 429:
            retry_after = None
            if isinstance(data, dict) and "retry_after" in data:
                retry_after = data["retry_after"]
            raise RateLimitError(error_msg, retry_after)
        elif status_code == 503:
            raise ServiceUnavailableError(error_msg)
        else:
            raise APIError(error_msg, status_code)

    def _create_get_function(
        self, endpoint: str, params: Optional[Dict[str, Any]], headers: Optional[Dict[str, str]]
    ) -> Callable[[], Awaitable[Dict[str, Any]]]:
        """
        Create an async function that makes a GET request.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            headers: Additional headers for this request

        Returns:
            Async function that makes the GET request
        """

        async def _get_with_retry():
            if self.session is None or self.session.closed:
                raise RuntimeError(
                    "Client session not initialized. Use 'async with' context manager."
                )

            url = self._build_url(endpoint)
            merged_headers = {**self.headers, **(headers or {})}

            logger.debug(f"GET {url} with params: {params}")

            try:
                async with self.session.get(
                    url=url, params=params, headers=merged_headers
                ) as response:
                    return await self._handle_response(response)
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error in GET {url}: {str(e)}")
                raise APIError(f"HTTP request failed: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout in GET {url}")
                raise APITimeoutError(f"Request to {url} timed out after {self.timeout}s")

        return _get_with_retry

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a GET request to the API with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            headers: Additional headers for this request

        Returns:
            JSON response data

        Raises:
            APIError: If the request fails after retries
        """
        # Use the retry decorator created during initialization
        return await self._retry_decorator(self._create_get_function(endpoint, params, headers))()

    async def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request to the API with retry logic.

        Args:
            endpoint: API endpoint path
            json: JSON body data
            params: Query parameters
            headers: Additional headers for this request

        Returns:
            JSON response data

        Raises:
            APIError: If the request fails after retries
        """
        # Use the retry decorator created during initialization
        return await self._retry_decorator(
            self._create_post_function(endpoint, json, params, headers)
        )()

    def _create_post_function(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
    ) -> Callable[[], Awaitable[Dict[str, Any]]]:
        """
        Create an async function that makes a POST request.

        Args:
            endpoint: API endpoint path
            json: JSON body data
            params: Query parameters
            headers: Additional headers for this request

        Returns:
            Async function that makes the POST request
        """

        async def _post_with_retry():
            if self.session is None or self.session.closed:
                raise RuntimeError(
                    "Client session not initialized. Use 'async with' context manager."
                )

            url = self._build_url(endpoint)
            merged_headers = {**self.headers, **(headers or {})}

            logger.debug(f"POST {url} with params: {params}, json: {json}")

            try:
                async with self.session.post(
                    url=url, json=json, params=params, headers=merged_headers
                ) as response:
                    return await self._handle_response(response)
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error in POST {url}: {str(e)}")
                raise APIError(f"HTTP request failed: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout in POST {url}")
                raise APITimeoutError(f"Request to {url} timed out after {self.timeout}s")

        return _post_with_retry
