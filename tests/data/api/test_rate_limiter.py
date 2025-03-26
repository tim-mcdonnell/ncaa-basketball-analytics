import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Import the rate limiter that doesn't exist yet (will implement based on this test)
from src.data.api.rate_limiter import AdaptiveRateLimiter


class TestAdaptiveRateLimiter:
    """Tests for the adaptive rate limiter implementation."""

    def test_initialization(self):
        """Test rate limiter initializes with correct configuration."""
        # Arrange & Act
        limiter = AdaptiveRateLimiter(initial=10, min_limit=5, max_limit=20)
        
        # Assert
        assert limiter.current_limit == 10
        assert limiter.min_limit == 5
        assert limiter.max_limit == 20
        assert limiter.semaphore._value == 10
        assert limiter.success_streak == 0
        assert limiter.failure_streak == 0
    
    @pytest.mark.asyncio
    async def test_acquire_release_success(self):
        """Test semaphore acquire and release with success."""
        # Arrange
        limiter = AdaptiveRateLimiter(initial=2)
        
        # Act & Assert
        # Should be able to acquire twice without blocking
        await limiter.acquire()
        assert limiter.semaphore._value == 1
        
        await limiter.acquire()
        assert limiter.semaphore._value == 0
        
        # Release with success should increment success streak
        await limiter.release(success=True)
        assert limiter.semaphore._value == 1
        assert limiter.success_streak == 1
        assert limiter.failure_streak == 0
    
    @pytest.mark.asyncio
    async def test_acquire_release_failure(self):
        """Test semaphore acquire and release with failure."""
        # Arrange
        limiter = AdaptiveRateLimiter(initial=2)
        
        # Act
        await limiter.acquire()
        await limiter.release(success=False)
        
        # Assert
        assert limiter.success_streak == 0
        assert limiter.failure_streak == 1
    
    @pytest.mark.asyncio
    async def test_increase_concurrency_after_success_streak(self):
        """Test concurrency increases after consecutive successes."""
        # Arrange
        limiter = AdaptiveRateLimiter(initial=5, max_limit=10)
        limiter.success_streak = 9  # One less than the threshold
        
        # Act
        await limiter.release(success=True)  # This should trigger an increase
        
        # Assert
        assert limiter.current_limit == 6  # Should increase by 1
        # The semaphore implementation details may vary, so we only check the limit
        assert limiter.success_streak == 10
    
    @pytest.mark.asyncio
    async def test_decrease_concurrency_after_failure_streak(self):
        """Test concurrency decreases after consecutive failures."""
        # Arrange
        limiter = AdaptiveRateLimiter(initial=5, min_limit=2)
        limiter.failure_streak = 2  # One less than the threshold
        
        # Act
        await limiter.release(success=False)  # This should trigger a decrease
        
        # Assert
        assert limiter.current_limit == 4  # Should decrease by 1
        # The semaphore implementation details may vary, so we only check the limit
        assert limiter.failure_streak == 3
    
    @pytest.mark.asyncio
    async def test_concurrency_respects_limits(self):
        """Test concurrency stays within min and max limits."""
        # Arrange
        limiter = AdaptiveRateLimiter(initial=2, min_limit=2, max_limit=5)
        
        # Act & Assert - Test min limit
        limiter.failure_streak = 3
        await limiter.release(success=False)  # Try to decrease below min
        assert limiter.current_limit == 2  # Should not go below min
        
        # Act & Assert - Test max limit
        limiter = AdaptiveRateLimiter(initial=5, min_limit=2, max_limit=5)
        limiter.success_streak = 10
        await limiter.release(success=True)  # Try to increase above max
        assert limiter.current_limit == 5  # Should not go above max 