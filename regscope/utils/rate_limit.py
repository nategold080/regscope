"""Interval-based rate limiter for API requests."""

import logging
import time
import threading

logger = logging.getLogger(__name__)


class IntervalRateLimiter:
    """Rate limiter using minimum-interval enforcement with API header tracking.

    Proactively throttles requests based on the X-RateLimit-Remaining header
    from the Regulations.gov API. Does NOT just catch 429s — sleeps before
    hitting the limit.

    Args:
        requests_per_hour: Maximum requests allowed per hour (default 1000).
        target_per_hour: Target rate to leave buffer (default 900).
    """

    def __init__(self, requests_per_hour: int = 1000, target_per_hour: int = 900) -> None:
        self.max_per_hour = requests_per_hour
        self.target_per_hour = target_per_hour
        self.min_interval = 3600.0 / target_per_hour  # seconds between requests
        self.remaining: int | None = None
        self.reset_time: float | None = None
        self.last_request_time: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until it's safe to make another request.

        Uses the minimum interval between requests, plus additional sleeping
        if the API reports low remaining quota.
        """
        with self._lock:
            now = time.time()

            # If we know remaining count is low, sleep longer
            if self.remaining is not None and self.remaining < 50:
                sleep_time = max(60.0, self.min_interval * 3)
                logger.warning(
                    "Rate limit nearly exhausted (%d remaining). Sleeping %.1fs",
                    self.remaining,
                    sleep_time,
                )
                time.sleep(sleep_time)
                return

            # Enforce minimum interval between requests
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug("Rate limiting: sleeping %.2fs", sleep_time)
                time.sleep(sleep_time)

            self.last_request_time = time.time()

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update rate limit state from API response headers.

        Args:
            headers: Response headers dictionary.
        """
        with self._lock:
            remaining = headers.get("X-RateLimit-Remaining")
            if remaining is not None:
                try:
                    self.remaining = int(remaining)
                    logger.debug("Rate limit remaining: %d", self.remaining)
                except ValueError:
                    pass

    def handle_429(self, retry_after: str | None = None) -> None:
        """Handle a 429 Too Many Requests response.

        Args:
            retry_after: Value of Retry-After header, if present.
        """
        with self._lock:
            if retry_after:
                try:
                    sleep_time = min(float(retry_after), 600.0)  # Cap at 10 minutes
                except ValueError:
                    sleep_time = 60.0
            else:
                sleep_time = 60.0

            logger.warning("Got 429 rate limited. Sleeping %.0fs", sleep_time)
            self.remaining = 0

        # Sleep outside the lock to avoid blocking other threads
        time.sleep(sleep_time)


# Backward-compatible alias
TokenBucketRateLimiter = IntervalRateLimiter
