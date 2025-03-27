import pytest
from unittest.mock import patch, AsyncMock

from src.data.api.espn_client import AsyncESPNClient
from src.data.api.exceptions import ServiceUnavailableError, ConnectionResetError


class TestAsyncESPNClientRecovery:
    """Tests for the enhanced recovery mechanisms in the ESPN API client."""

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.client.AsyncESPNClient.get")
    async def test_get_with_enhanced_recovery_success(self, mock_get):
        """Test that enhanced recovery works successfully after intermittent errors."""
        # Arrange
        mock_get.side_effect = [
            ServiceUnavailableError("Service temporarily unavailable"),  # First attempt fails
            {"data": "success"},  # Second attempt succeeds
        ]
        client = AsyncESPNClient()

        # Act
        result = await client.get_with_enhanced_recovery("/test-endpoint")

        # Assert
        assert result == {"data": "success"}
        assert mock_get.call_count == 2

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.client.AsyncESPNClient.get")
    @patch("asyncio.sleep", new_callable=AsyncMock)  # Mock sleep to avoid waiting
    async def test_get_with_enhanced_recovery_connection_reset(self, mock_sleep, mock_get):
        """Test that enhanced recovery works for connection reset errors."""
        # Arrange
        mock_get.side_effect = [
            ConnectionResetError("Connection reset by peer"),  # First attempt fails
            ConnectionResetError("Connection reset by peer"),  # Second attempt fails
            {"data": "success"},  # Third attempt succeeds
        ]
        client = AsyncESPNClient()

        # Act
        result = await client.get_with_enhanced_recovery("/test-endpoint", max_recovery_attempts=3)

        # Assert
        assert result == {"data": "success"}
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # Should sleep between attempts

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.client.AsyncESPNClient.get")
    @patch("asyncio.sleep", new_callable=AsyncMock)  # Mock sleep to avoid waiting
    async def test_get_with_enhanced_recovery_max_attempts(self, mock_sleep, mock_get):
        """Test that enhanced recovery gives up after max attempts."""
        # Arrange
        error = ServiceUnavailableError("Service temporarily unavailable")
        mock_get.side_effect = [error, error, error]  # All attempts fail
        client = AsyncESPNClient()

        # Act & Assert
        with pytest.raises(ServiceUnavailableError):
            await client.get_with_enhanced_recovery("/test-endpoint", max_recovery_attempts=2)

        # The client attempts the original call plus recovery_attempts
        assert mock_get.call_count == 3  # Initial attempt + 2 recovery attempts

        # Should sleep once per recovery attempt (it should make 3 total attempts,
        # but sleep only happens before the 2nd and 3rd attempts)
        assert mock_sleep.call_count == 3  # Should sleep after each failed attempt

    @pytest.mark.asyncio
    @patch("src.data.api.espn_client.client.AsyncESPNClient.get")
    async def test_get_with_enhanced_recovery_other_error(self, mock_get):
        """Test that enhanced recovery doesn't retry for non-recoverable errors."""
        # Arrange
        mock_get.side_effect = ValueError("Some other error")
        client = AsyncESPNClient()

        # Act & Assert
        with pytest.raises(ValueError):
            await client.get_with_enhanced_recovery("/test-endpoint")

        assert mock_get.call_count == 1  # Should not retry for non-recoverable errors
