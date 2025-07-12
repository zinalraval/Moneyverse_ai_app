import time
from collections import deque
from typing import Deque, Tuple
import logging
import asyncio

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def check_rate_limit(self) -> bool:
        """
        Check if a request is allowed under the rate limit.
        
        Returns:
            bool: True if request is allowed, False if rate limit exceeded
        """
        async with self._lock:
            now = time.time()
            
            # Remove expired timestamps
            while self.requests and now - self.requests[0] > self.window_seconds:
                self.requests.popleft()
            
            # Check if we're under the limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            logger.warning("Rate limit exceeded")
            return False

    def get_current_usage(self) -> Tuple[int, int]:
        """
        Get current rate limit usage.
        
        Returns:
            Tuple[int, int]: (current_requests, max_requests)
        """
        now = time.time()
        while self.requests and now - self.requests[0] > self.window_seconds:
            self.requests.popleft()
        
        return len(self.requests), self.max_requests 