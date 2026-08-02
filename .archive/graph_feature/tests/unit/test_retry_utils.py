"""
Unit tests for graph_processing/retry_utils.py.

Covers:
- is_rate_limit_error
- is_timeout_error
- is_server_error
- should_retry
- calculate_delay (exponential backoff, rate limit, jitter)
- retry_with_backoff decorator
- retry_async_with_backoff decorator
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from graph_processing.retry_utils import (
    is_rate_limit_error,
    is_timeout_error,
    is_server_error,
    should_retry,
    calculate_delay,
    retry_with_backoff,
    retry_async_with_backoff,
    RateLimitError,
)


class TestIsRateLimitError:
    @pytest.mark.parametrize("msg", [
        "Rate limit exceeded",
        "Quota exceeded for project",
        "Resource exhausted",
        "HTTP 429 Too Many Requests",
        "Too many requests, slow down",
        "Requests per minute limit reached",
        "rate_limit error",
        "ratelimit hit",
    ])
    def test_rate_limit_messages(self, msg):
        assert is_rate_limit_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", [
        "Connection refused",
        "Not found",
        "Internal error",
    ])
    def test_non_rate_limit_messages(self, msg):
        assert is_rate_limit_error(Exception(msg)) is False


class TestIsTimeoutError:
    @pytest.mark.parametrize("msg", [
        "Connection timeout",
        "Request timed out",
        "Deadline exceeded",
    ])
    def test_timeout_messages(self, msg):
        assert is_timeout_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", [
        "Connection refused",
        "Not found",
    ])
    def test_non_timeout_messages(self, msg):
        assert is_timeout_error(Exception(msg)) is False


class TestIsServerError:
    @pytest.mark.parametrize("msg", [
        "HTTP 500 Internal Server Error",
        "502 Bad Gateway",
        "503 Service Unavailable",
        "504 Gateway Timeout",
        "Internal server error occurred",
        "Bad gateway detected",
        "Service unavailable right now",
        "Gateway timeout reached",
    ])
    def test_server_error_messages(self, msg):
        assert is_server_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", [
        "400 Bad Request",
        "404 Not Found",
        "Connection refused",
    ])
    def test_non_server_error_messages(self, msg):
        assert is_server_error(Exception(msg)) is False


class TestShouldRetry:
    def test_retries_on_rate_limit(self):
        assert should_retry(Exception("rate limit"), 0, 3) is True

    def test_retries_on_timeout(self):
        assert should_retry(Exception("timeout"), 0, 3) is True

    def test_retries_on_server_error(self):
        assert should_retry(Exception("503 service unavailable"), 0, 3) is True

    def test_no_retry_on_non_retryable_error(self):
        assert should_retry(Exception("bad request"), 0, 3) is False

    def test_no_retry_when_max_retries_exceeded(self):
        assert should_retry(Exception("rate limit"), 3, 3) is False

    def test_retries_when_below_max(self):
        assert should_retry(Exception("rate limit"), 2, 3) is True


class TestCalculateDelay:
    def test_exponential_backoff_no_jitter(self):
        delay = calculate_delay(
            retry_count=0, initial_delay=2.0, max_delay=60.0,
            exponential_base=2.0, add_jitter=False
        )
        assert delay == 2.0

    def test_exponential_backoff_increases(self):
        delay0 = calculate_delay(0, 2.0, 60.0, 2.0, add_jitter=False)
        delay1 = calculate_delay(1, 2.0, 60.0, 2.0, add_jitter=False)
        delay2 = calculate_delay(2, 2.0, 60.0, 2.0, add_jitter=False)
        assert delay0 < delay1 < delay2

    def test_capped_at_max_delay(self):
        delay = calculate_delay(
            retry_count=10, initial_delay=2.0, max_delay=60.0,
            exponential_base=2.0, add_jitter=False
        )
        assert delay == 60.0

    def test_rate_limit_uses_special_pause(self):
        delay = calculate_delay(
            retry_count=0, initial_delay=2.0, max_delay=60.0,
            exponential_base=2.0, is_rate_limit=True,
            rate_limit_pause=65.0, add_jitter=False
        )
        assert delay == 65.0

    def test_jitter_adds_variability(self):
        delays = set()
        for _ in range(20):
            d = calculate_delay(
                retry_count=1, initial_delay=2.0, max_delay=60.0,
                exponential_base=2.0, add_jitter=True
            )
            delays.add(round(d, 4))
        assert len(delays) > 1


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1, exponential_base=2.0)
        def succeed():
            return "ok"

        assert succeed() == "ok"

    @patch('graph_processing.retry_utils.time.sleep')
    def test_retries_then_succeeds(self, mock_sleep):
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1, exponential_base=2.0)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limit exceeded")
            return "ok"

        result = fail_then_succeed()
        assert result == "ok"
        assert call_count == 3

    @patch('graph_processing.retry_utils.time.sleep')
    def test_raises_after_max_retries(self, mock_sleep):
        @retry_with_backoff(max_retries=2, initial_delay=0.01, max_delay=0.1, exponential_base=2.0)
        def always_fail():
            raise Exception("rate limit exceeded")

        with pytest.raises(Exception, match="rate limit"):
            always_fail()

    def test_non_retryable_error_raises_immediately(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1, exponential_base=2.0)
        def non_retryable():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            non_retryable()
        assert call_count == 1


class TestRetryAsyncWithBackoff:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        @retry_async_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1, exponential_base=2.0)
        async def succeed():
            return "ok"

        result = await succeed()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        call_count = 0

        @retry_async_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1, exponential_base=2.0)
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("timeout exceeded")
            return "ok"

        result = await fail_then_succeed()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        @retry_async_with_backoff(max_retries=2, initial_delay=0.01, max_delay=0.1, exponential_base=2.0)
        async def always_fail():
            raise Exception("503 service unavailable")

        with pytest.raises(Exception, match="503"):
            await always_fail()
