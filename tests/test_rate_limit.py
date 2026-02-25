"""Tests for regscope.utils.rate_limit module."""

import threading
import time
from unittest.mock import patch, call

import pytest

from regscope.utils.rate_limit import IntervalRateLimiter, TokenBucketRateLimiter


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

class TestIntervalRateLimiterInit:
    """Verify that the constructor computes correct defaults."""

    def test_default_min_interval(self) -> None:
        """Default target of 900 req/hr should give min_interval = 4.0s."""
        limiter = IntervalRateLimiter()
        assert limiter.min_interval == pytest.approx(3600.0 / 900)

    def test_custom_target(self) -> None:
        """A custom target_per_hour should update min_interval accordingly."""
        limiter = IntervalRateLimiter(requests_per_hour=500, target_per_hour=400)
        assert limiter.min_interval == pytest.approx(3600.0 / 400)

    def test_initial_remaining_is_none(self) -> None:
        """Before any API response, remaining should be None."""
        limiter = IntervalRateLimiter()
        assert limiter.remaining is None

    def test_initial_last_request_time_is_zero(self) -> None:
        """Before any call, last_request_time should be 0.0."""
        limiter = IntervalRateLimiter()
        assert limiter.last_request_time == 0.0


# ---------------------------------------------------------------------------
# Minimum interval enforcement
# ---------------------------------------------------------------------------

class TestWaitMinInterval:
    """wait() should sleep to enforce min_interval between consecutive calls."""

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    @patch("regscope.utils.rate_limit.time.time")
    def test_two_rapid_calls_cause_sleep(self, mock_time, mock_sleep) -> None:
        """Two calls in rapid succession should trigger a sleep for the remaining
        interval duration."""
        limiter = IntervalRateLimiter()  # min_interval = 4.0s

        # First call: time returns 1000.0, elapsed since 0.0 = 1000s (well past
        # min_interval), so no sleep needed.  After wait(), last_request_time is
        # set to current time.
        mock_time.return_value = 1000.0
        limiter.wait()
        mock_sleep.assert_not_called()
        assert limiter.last_request_time == 1000.0

        # Second call: only 1.0s later — should sleep for ~3.0s
        mock_time.return_value = 1001.0
        limiter.wait()
        expected_sleep = limiter.min_interval - 1.0  # 4.0 - 1.0 = 3.0
        mock_sleep.assert_called_once()
        actual_sleep = mock_sleep.call_args[0][0]
        assert actual_sleep == pytest.approx(expected_sleep)

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    @patch("regscope.utils.rate_limit.time.time")
    def test_no_sleep_when_interval_exceeded(self, mock_time, mock_sleep) -> None:
        """If enough time has passed since the last request, no sleep should occur."""
        limiter = IntervalRateLimiter()
        # First call far in the past
        mock_time.return_value = 1000.0
        limiter.wait()
        # Second call well after min_interval
        mock_time.return_value = 1010.0  # 10s later, min_interval is 4s
        limiter.wait()
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Low remaining quota behaviour
# ---------------------------------------------------------------------------

class TestWaitLowRemaining:
    """When remaining < 50, wait() should use extended sleep regardless of
    elapsed time."""

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    @patch("regscope.utils.rate_limit.time.time")
    def test_low_remaining_triggers_extended_sleep(self, mock_time, mock_sleep) -> None:
        """With remaining=10, wait() should sleep at least 60s (or 3x min_interval,
        whichever is greater)."""
        limiter = IntervalRateLimiter()
        limiter.remaining = 10
        mock_time.return_value = 99999.0  # Large elapsed — irrelevant
        limiter.wait()

        expected_sleep = max(60.0, limiter.min_interval * 3)
        mock_sleep.assert_called_once_with(expected_sleep)

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    @patch("regscope.utils.rate_limit.time.time")
    def test_remaining_at_zero(self, mock_time, mock_sleep) -> None:
        """remaining=0 (e.g. after 429) should also trigger extended sleep."""
        limiter = IntervalRateLimiter()
        limiter.remaining = 0
        mock_time.return_value = 99999.0
        limiter.wait()
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] >= 60.0

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    @patch("regscope.utils.rate_limit.time.time")
    def test_remaining_at_boundary_50_does_not_trigger_extended(
        self, mock_time, mock_sleep
    ) -> None:
        """remaining=50 is NOT below the threshold, so normal interval applies."""
        limiter = IntervalRateLimiter()
        limiter.remaining = 50
        # Set up time so no normal interval sleep is needed either
        mock_time.return_value = 99999.0
        limiter.last_request_time = 0.0
        limiter.wait()
        # Should NOT have slept (elapsed is massive)
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# update_from_headers
# ---------------------------------------------------------------------------

class TestUpdateFromHeaders:
    """update_from_headers should parse X-RateLimit-Remaining from headers."""

    def test_sets_remaining_from_header(self) -> None:
        """Valid integer header value should update self.remaining."""
        limiter = IntervalRateLimiter()
        limiter.update_from_headers({"X-RateLimit-Remaining": "742"})
        assert limiter.remaining == 742

    def test_missing_header_leaves_remaining_unchanged(self) -> None:
        """If header is absent, remaining should stay at its prior value."""
        limiter = IntervalRateLimiter()
        limiter.remaining = 500
        limiter.update_from_headers({"Content-Type": "application/json"})
        assert limiter.remaining == 500

    def test_non_numeric_header_leaves_remaining_unchanged(self) -> None:
        """A non-parseable value should be silently ignored."""
        limiter = IntervalRateLimiter()
        limiter.remaining = 200
        limiter.update_from_headers({"X-RateLimit-Remaining": "not-a-number"})
        assert limiter.remaining == 200

    def test_zero_remaining(self) -> None:
        """Header value '0' should set remaining to 0."""
        limiter = IntervalRateLimiter()
        limiter.update_from_headers({"X-RateLimit-Remaining": "0"})
        assert limiter.remaining == 0

    def test_case_sensitive_header_name(self) -> None:
        """Standard dict lookup is case-sensitive; lowercase key should not match."""
        limiter = IntervalRateLimiter()
        limiter.update_from_headers({"x-ratelimit-remaining": "100"})
        # Standard dict won't match the mixed-case key
        assert limiter.remaining is None


# ---------------------------------------------------------------------------
# handle_429
# ---------------------------------------------------------------------------

class TestHandle429:
    """handle_429 should set remaining=0 and sleep appropriately."""

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    def test_with_retry_after_header(self, mock_sleep) -> None:
        """Should sleep for the specified Retry-After duration."""
        limiter = IntervalRateLimiter()
        limiter.remaining = 100
        limiter.handle_429(retry_after="120")
        assert limiter.remaining == 0
        mock_sleep.assert_called_once_with(120.0)

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    def test_without_retry_after_defaults_to_60(self, mock_sleep) -> None:
        """Without Retry-After, should sleep for the default 60 seconds."""
        limiter = IntervalRateLimiter()
        limiter.handle_429(retry_after=None)
        assert limiter.remaining == 0
        mock_sleep.assert_called_once_with(60.0)

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    def test_retry_after_capped_at_600(self, mock_sleep) -> None:
        """Retry-After values above 600 should be capped at 600 seconds."""
        limiter = IntervalRateLimiter()
        limiter.handle_429(retry_after="9999")
        mock_sleep.assert_called_once_with(600.0)

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    def test_non_numeric_retry_after_defaults_to_60(self, mock_sleep) -> None:
        """Unparseable Retry-After should fall back to 60s default."""
        limiter = IntervalRateLimiter()
        limiter.handle_429(retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
        mock_sleep.assert_called_once_with(60.0)

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    def test_retry_after_float_value(self, mock_sleep) -> None:
        """Retry-After with a decimal value should be parsed as float."""
        limiter = IntervalRateLimiter()
        limiter.handle_429(retry_after="30.5")
        mock_sleep.assert_called_once_with(30.5)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """TokenBucketRateLimiter should be a working alias for IntervalRateLimiter."""

    def test_alias_is_same_class(self) -> None:
        """The alias should point to the exact same class object."""
        assert TokenBucketRateLimiter is IntervalRateLimiter

    def test_alias_creates_functional_instance(self) -> None:
        """Instantiating through the alias should produce a fully functional limiter."""
        limiter = TokenBucketRateLimiter(requests_per_hour=500, target_per_hour=250)
        assert limiter.min_interval == pytest.approx(3600.0 / 250)
        assert isinstance(limiter, IntervalRateLimiter)

    def test_isinstance_check(self) -> None:
        """An instance created via the alias should pass isinstance checks for both names."""
        limiter = TokenBucketRateLimiter()
        assert isinstance(limiter, TokenBucketRateLimiter)
        assert isinstance(limiter, IntervalRateLimiter)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Verify that concurrent access to the rate limiter doesn't corrupt state."""

    def test_has_lock_attribute(self) -> None:
        """The limiter should have a threading lock for internal synchronization."""
        limiter = IntervalRateLimiter()
        assert hasattr(limiter, "_lock")
        assert isinstance(limiter._lock, type(threading.Lock()))

    def test_concurrent_update_from_headers(self) -> None:
        """Multiple threads calling update_from_headers should not corrupt remaining.

        After all threads complete, remaining should hold one of the written values
        (not a garbled intermediate).
        """
        limiter = IntervalRateLimiter()
        values_written = list(range(100))
        errors: list[Exception] = []

        def update(val: int) -> None:
            try:
                limiter.update_from_headers({"X-RateLimit-Remaining": str(val)})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update, args=(v,)) for v in values_written]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Threads raised exceptions: {errors}"
        # Final value must be one of the values we wrote — no corruption
        assert limiter.remaining in values_written

    @patch("regscope.utils.rate_limit.time.sleep", return_value=None)
    @patch("regscope.utils.rate_limit.time.time", return_value=99999.0)
    def test_concurrent_wait_calls(self, mock_time, mock_sleep) -> None:
        """Multiple threads calling wait() should not raise exceptions."""
        limiter = IntervalRateLimiter()
        errors: list[Exception] = []

        def do_wait() -> None:
            try:
                limiter.wait()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=do_wait) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Threads raised exceptions: {errors}"
