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
    When the limit is reached, ``acquire()`` blocks until a slot frees up.
    Accepts an optional async callback to notify the user before waiting.
    """

    def __init__(self, rpm: int = 0):
        """
        Args:
            rpm: Maximum requests per minute. 0 = unlimited.
        """
        self._rpm = rpm
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        logger.info("MinuteRateLimiter created", rpm=rpm)

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

    async def acquire(self, notify_wait=None) -> float:
        """
        Acquire a slot.  Blocks if rate limit is reached.

        Args:
            notify_wait: Optional async callback ``async def(seconds: float)``
                called once before the first wait, so the caller can notify
                the user.  Called **outside** the lock.

        Returns:
            Total wait time in seconds (0.0 if slot was immediately available).
        """
        if not self.is_enabled:
            return 0.0

        total_wait = 0.0
        notified = False

        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge_old(now)

                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    logger.debug(
                        "Rate limiter: slot acquired",
                        used=len(self._timestamps),
                        limit=self._rpm,
                    )
                    return total_wait

                # Calculate wait time while holding the lock
                wait_until = self._timestamps[0] + 60.0
                wait_seconds = max(wait_until - now, 0.1)

            # Lock released — notify user and sleep
            logger.info(
                "Rate limit reached, waiting",
                used=len(self._timestamps),
                limit=self._rpm,
                wait_seconds=round(wait_seconds, 1),
            )

            if notify_wait and not notified:
                await notify_wait(wait_seconds)
                notified = True

            await asyncio.sleep(wait_seconds)
            total_wait += wait_seconds
