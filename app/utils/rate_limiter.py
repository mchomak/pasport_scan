"""Rate limiter for API calls."""
import asyncio
import time
from typing import Optional


class RateLimiter:
    """Async rate limiter using token bucket algorithm."""

    def __init__(self, rate: float):
        """
        Initialize rate limiter.

        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.interval = 1.0 / rate if rate > 0 else 0
        self.last_call: Optional[float] = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self.rate <= 0:
            return

        async with self._lock:
            now = time.monotonic()

            if self.last_call is not None:
                elapsed = now - self.last_call
                sleep_time = self.interval - elapsed

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    now = time.monotonic()

            self.last_call = now
