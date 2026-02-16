"""Regulations.gov API v4 client."""

import logging
import time
from typing import Any

import httpx

from regscope.utils.rate_limit import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

BASE_URL = "https://api.regulations.gov/v4"


class RegulationsClient:
    """Client for the Regulations.gov API v4.

    Handles authentication, rate limiting, pagination, retries, and
    the two-phase comment download process. Supports multiple API keys
    with round-robin rotation for higher throughput.

    Args:
        api_key: Regulations.gov API key (or comma-separated list of keys).
        config: Application configuration dictionary.
    """

    def __init__(self, api_key: str, config: dict[str, Any]) -> None:
        api_cfg = config.get("api", {})
        # Support multiple keys separated by commas
        self.api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        if not self.api_keys:
            self.api_keys = [api_key]
        self._key_index = 0
        self.max_retries = api_cfg.get("max_retries", 3)
        self.backoff_base = api_cfg.get("retry_backoff_base", 2.0)
        # Per-key rate limiters
        target = api_cfg.get("requests_per_hour", 900)
        self._rate_limiters = {
            key: TokenBucketRateLimiter(requests_per_hour=1000, target_per_hour=target)
            for key in self.api_keys
        }
        self.client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )
        if len(self.api_keys) > 1:
            logger.info("Using %d API keys with round-robin rotation", len(self.api_keys))

    def _next_key(self) -> str:
        """Get the next API key in the rotation."""
        key = self.api_keys[self._key_index % len(self.api_keys)]
        self._key_index += 1
        return key

    def _request(self, method: str, url: str, params: dict[str, Any] | None = None) -> dict | None:
        """Make a rate-limited, retrying request to the API.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Full URL to request.
            params: Query parameters.

        Returns:
            Parsed JSON response, or None on failure.
        """
        if params is None:
            params = {}
        api_key = self._next_key()
        rate_limiter = self._rate_limiters[api_key]
        params["api_key"] = api_key

        for attempt in range(self.max_retries + 1):
            rate_limiter.wait()

            try:
                response = self.client.request(method, url, params=params)
                rate_limiter.update_from_headers(dict(response.headers))

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    rate_limiter.handle_429(retry_after)
                    continue

                if response.status_code == 404:
                    logger.warning("404 Not Found: %s", url)
                    return None

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException:
                logger.warning("Request timeout (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, url)
            except httpx.HTTPStatusError as e:
                logger.warning("HTTP error %d (attempt %d/%d): %s", e.response.status_code, attempt + 1, self.max_retries + 1, url)
            except httpx.RequestError as e:
                logger.warning("Request error (attempt %d/%d): %s — %s", attempt + 1, self.max_retries + 1, url, str(e))

            if attempt < self.max_retries:
                sleep_time = self.backoff_base ** (attempt + 1)
                logger.info("Retrying in %.1fs...", sleep_time)
                time.sleep(sleep_time)

        logger.error("All retries exhausted for %s", url)
        return None

    def get_docket(self, docket_id: str) -> dict | None:
        """Fetch docket metadata.

        Args:
            docket_id: The docket ID (e.g., 'EPA-HQ-OAR-2021-0317').

        Returns:
            Docket data dict with 'id' and 'attributes', or None.
        """
        url = f"{BASE_URL}/dockets/{docket_id}"
        result = self._request("GET", url)
        if result:
            return result.get("data")
        return None

    def list_comments(
        self,
        docket_id: str,
        page: int = 1,
        page_size: int = 250,
        last_modified_date: str | None = None,
    ) -> list[dict]:
        """List comments for a docket via the comments endpoint.

        Args:
            docket_id: The docket ID.
            page: Page number (1-indexed, max 20).
            page_size: Number of results per page (max 250).
            last_modified_date: Filter for cursor-based pagination.

        Returns:
            List of comment data dicts.
        """
        params: dict[str, Any] = {
            "filter[docketId]": docket_id,
            "page[size]": min(page_size, 250),
            "page[number]": page,
            "sort": "lastModifiedDate",
        }

        if last_modified_date:
            params["filter[lastModifiedDate][ge]"] = last_modified_date

        result = self._request("GET", f"{BASE_URL}/comments", params=params)
        if result:
            return result.get("data", [])
        return []

    def get_comment(self, comment_id: str) -> dict | None:
        """Fetch full detail for a single comment.

        Args:
            comment_id: The comment ID.

        Returns:
            Comment data dict with 'id', 'attributes', and optionally 'included'
            (attachments), or None.
        """
        params = {"include": "attachments"}
        result = self._request("GET", f"{BASE_URL}/comments/{comment_id}", params=params)
        if result:
            data = result.get("data", {})
            data["included"] = result.get("included", [])
            return data
        return None

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
