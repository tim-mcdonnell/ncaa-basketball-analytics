import asyncio
import logging

logger = logging.getLogger(__name__)

class AdaptiveRateLimiter:
    """
    Dynamically adjusts concurrency levels for API requests based on success/failure patterns.
    
    This limiter uses a semaphore to control concurrent access and adaptively adjusts
    its limit based on the observed success and failure patterns of API requests.
    """
    
    def __init__(
        self,
        initial: int = 10,
        min_limit: int = 1,
        max_limit: int = 50,
        success_threshold: int = 10,
        failure_threshold: int = 3
    ):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            initial: Initial concurrency limit
            min_limit: Minimum concurrency limit
            max_limit: Maximum concurrency limit
            success_threshold: Number of consecutive successes before increasing limit
            failure_threshold: Number of consecutive failures before decreasing limit
        """
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.current_limit = initial
        self.semaphore = asyncio.Semaphore(initial)
        self.success_streak = 0
        self.failure_streak = 0
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """
        Acquire the semaphore before making a request.
        This will block if the concurrency limit has been reached.
        """
        await self.semaphore.acquire()
    
    async def release(self, success: bool = True):
        """
        Release the semaphore after a request completes.
        Updates success/failure streaks and adjusts concurrency if needed.
        
        Args:
            success: Whether the request was successful
        """
        self.semaphore.release()
        
        async with self.lock:
            if success:
                self.success_streak += 1
                self.failure_streak = 0
                
                # Increase concurrency after consecutive successes
                if self.success_streak >= self.success_threshold and self.current_limit < self.max_limit:
                    await self._increase_concurrency()
            else:
                self.failure_streak += 1
                self.success_streak = 0
                
                # Decrease concurrency after consecutive failures
                if self.failure_streak >= self.failure_threshold and self.current_limit > self.min_limit:
                    await self._decrease_concurrency()
    
    async def _increase_concurrency(self):
        """Increase concurrency limit by 1."""
        # Determine the new limit
        new_limit = min(self.current_limit + 1, self.max_limit)
        
        # Only adjust if the limit would actually change
        if new_limit > self.current_limit:
            logger.info(f"Increasing concurrency limit from {self.current_limit} to {new_limit}")
            
            # Create a new semaphore with the new limit
            old_semaphore = self.semaphore
            self.semaphore = asyncio.Semaphore(new_limit)
            
            # Update the current limit
            self.current_limit = new_limit
            
            # Release additional permits to match the new semaphore's state
            for _ in range(new_limit - old_semaphore._value):
                self.semaphore.release()
    
    async def _decrease_concurrency(self):
        """Decrease concurrency limit by 1."""
        # Determine the new limit
        new_limit = max(self.current_limit - 1, self.min_limit)
        
        # Only adjust if the limit would actually change
        if new_limit < self.current_limit:
            logger.info(f"Decreasing concurrency limit from {self.current_limit} to {new_limit}")
            
            # Get the current available permits
            old_available = self.semaphore._value
            
            # In relative terms, how many permits are currently in use?
            permits_in_use = self.current_limit - old_available
            
            # Update the current limit
            self.current_limit = new_limit
            
            # Create a new semaphore
            self.semaphore = asyncio.Semaphore(new_limit)
            
            # Calculate how many permits should be available in the new semaphore
            # This should be (new_limit - permits_in_use), but ensure it's not negative
            new_available = max(new_limit - permits_in_use, 0)
            
            # Adjust the semaphore's available permits
            # First acquire all permits (semaphore starts with all available)
            for _ in range(new_limit):
                # We can use acquire_nowait() here because we know the permits are available
                # but we need to handle it differently for Python 3.12
                try:
                    acquire_task = self.semaphore.acquire()
                    await asyncio.wait_for(acquire_task, timeout=0.001)
                except asyncio.TimeoutError:
                    logger.warning("Unexpected timeout when acquiring semaphore")
                    break
            
            # Then release as many as should be available
            for _ in range(new_available):
                self.semaphore.release() 