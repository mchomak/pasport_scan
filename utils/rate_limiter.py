"""Rate limiters for API calls."""
import asyncio
import time
from collections import deque
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Async rate limiter using token bucket algorithm (per-second)."""

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


class MinuteRateLimiter:
    """
    Sliding-window rate limiter (per-minute), shared across all users.

    Tracks timestamps of requests within a 60-second window.
    When the limit is reached, ``acquire()`` blocks until a slot frees up
    and returns the total wait time so the caller can notify the user.
    """

    def __init__(self, rpm: int = 0):
        """
        Args:
            rpm: Maximum requests per minute. 0 = unlimited.
        """
        self._rpm = rpm
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    @property
    def rpm(self) -> int:
        return self._rpm

    @property
    def is_enabled(self) -> bool:
        return self._rpm > 0

    def _purge_old(self, now: float) -> None:
        cutoff = now - 60.0
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    async def acquire(self) -> float:
        """
        Acquire a slot.  Blocks if rate limit is reached.

        Returns:
            Wait time in seconds (0.0 if slot was immediately available).
        """
        if not self.is_enabled:
            return 0.0

        total_wait = 0.0

        async with self._lock:
            while True:
                now = time.monotonic()
                self._purge_old(now)

                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return total_wait

                wait_until = self._timestamps[0] + 60.0
                wait_seconds = max(wait_until - now, 0.1)

                logger.info(
                    "Rate limit reached, waiting",
                    current=len(self._timestamps),
                    limit=self._rpm,
                    wait_seconds=round(wait_seconds, 1),
                )

                # Release lock while sleeping so status checks don't deadlock
                self._lock.release()
                try:
                    await asyncio.sleep(wait_seconds)
                    total_wait += wait_seconds
                finally:
                    await self._lock.acquire()

    def remaining(self) -> int:
        """Requests still available in the current window (-1 if unlimited)."""
        if not self.is_enabled:
            return -1
        now = time.monotonic()
        self._purge_old(now)
        return max(self._rpm - len(self._timestamps), 0)

    def seconds_until_free(self) -> float:
        """Seconds until the next slot opens (0 if available now)."""
        if not self.is_enabled or not self._timestamps:
            return 0.0
        now = time.monotonic()
        self._purge_old(now)
        if len(self._timestamps) < self._rpm:
            return 0.0
        return max(self._timestamps[0] + 60.0 - now, 0.0)
