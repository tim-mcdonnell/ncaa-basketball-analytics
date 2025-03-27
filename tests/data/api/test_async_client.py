import pytest
from unittest.mock import patch, MagicMock

# Import the client that doesn't exist yet (will implement based on this test)
from src.data.api.async_client import AsyncClient


class TestAsyncClient:
    """Tests for the base asynchronous API client."""

    def test_initialization(self):
        """Test client initializes with correct configuration."""
        # Arrange
        base_url = "https://api.example.com"
        timeout = 30

        # Act
        client = AsyncClient(base_url=base_url, timeout=timeout)

        # Assert
        assert client.base_url == base_url
        assert client.timeout == timeout
        assert client.session is None  # Session should be created in __aenter__

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test client works as an async context manager."""
        # Arrange
        client = AsyncClient(base_url="https://api.example.com")

        # Act & Assert
        async with client as session_client:
            assert session_client.session is not None
            assert not session_client.session.closed

        # After context exit, session should be closed
        assert client.session.closed

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_get_request(self, mock_get):
        """Test GET request is properly formed and executed."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status = 200

        # Use AsyncMock for json method
        async def mock_json():
            return {"data": "test"}

        mock_response.json = mock_json
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncClient(base_url="https://api.example.com")

        # Act
        async with client:
            response = await client.get("/endpoint", params={"param": "value"})

        # Assert
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["url"] == "https://api.example.com/endpoint"
        assert kwargs["params"] == {"param": "value"}
        assert response == {"data": "test"}

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_get_request_error_handling(self, mock_get):
        """Test error handling in GET requests."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status = 404

        # Use AsyncMock for json method
        async def mock_json():
            return {"error": "Not found"}

        mock_response.json = mock_json
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncClient(base_url="https://api.example.com")

        # Act & Assert
        async with client:
            with pytest.raises(Exception):
                await client.get("/endpoint")
