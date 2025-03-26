import aiohttp
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class AsyncClient:
    """
    Base asynchronous HTTP client for API requests.
    Provides common functionality for making HTTP requests using aiohttp.
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the asynchronous client.
        
        Args:
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            headers: Optional default headers for requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers or {}
        self.session = None
    
    async def __aenter__(self):
        """Set up client session when entering async context."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self.headers
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
        if not endpoint.startswith('/'):
            endpoint = f'/{endpoint}'
        
        if self.base_url.endswith('/'):
            return f"{self.base_url[:-1]}{endpoint}"
        return f"{self.base_url}{endpoint}"
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            headers: Additional headers for this request
            
        Returns:
            JSON response data
            
        Raises:
            Exception: If the request fails or returns an error status
        """
        if self.session is None or self.session.closed:
            raise RuntimeError("Client session not initialized. Use 'async with' context manager.")
        
        url = self._build_url(endpoint)
        merged_headers = {**self.headers, **(headers or {})}
        
        logger.debug(f"GET {url} with params: {params}")
        
        async with self.session.get(url=url, params=params, headers=merged_headers) as response:
            data = await response.json()
            
            if response.status >= 400:
                logger.error(f"HTTP error {response.status}: {data}")
                raise Exception(f"HTTP error {response.status}: {data}")
            
            return data
    
    async def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint path
            json: JSON data for request body
            params: Query parameters
            headers: Additional headers for this request
            
        Returns:
            JSON response data
            
        Raises:
            Exception: If the request fails or returns an error status
        """
        if self.session is None or self.session.closed:
            raise RuntimeError("Client session not initialized. Use 'async with' context manager.")
        
        url = self._build_url(endpoint)
        merged_headers = {**self.headers, **(headers or {})}
        
        logger.debug(f"POST {url} with params: {params}, json: {json}")
        
        async with self.session.post(
            url=url, json=json, params=params, headers=merged_headers
        ) as response:
            data = await response.json()
            
            if response.status >= 400:
                logger.error(f"HTTP error {response.status}: {data}")
                raise Exception(f"HTTP error {response.status}: {data}")
            
            return data 