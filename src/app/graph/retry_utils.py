"""
Retry utilities for handling Gemini API rate limits and errors.

Features:
- Exponential backoff with jitter
- Special handling for rate limit errors (longer pause)
- Adaptive delay based on error type
"""

import time
import random
import logging
from typing import Callable, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error."""
    error_msg = str(exception).lower()
    rate_limit_indicators = [
        "rate limit",
        "quota exceeded",
        "resource exhausted",
        "429",
        "too many requests",
        "requests per",
        "rate_limit",
        "ratelimit",
    ]
    return any(indicator in error_msg for indicator in rate_limit_indicators)


def is_timeout_error(exception: Exception) -> bool:
    """Check if exception is a timeout error."""
    error_msg = str(exception).lower()
    timeout_indicators = [
        "timeout",
        "timed out",
        "deadline exceeded",
    ]
    return any(indicator in error_msg for indicator in timeout_indicators)


def is_server_error(exception: Exception) -> bool:
    """Check if exception is a server error (5xx)."""
    error_msg = str(exception).lower()
    server_error_indicators = [
        "500",
        "502",
        "503",
        "504",
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    ]
    return any(indicator in error_msg for indicator in server_error_indicators)


def should_retry(exception: Exception, retry_count: int, max_retries: int) -> bool:
    """
    Determine if an operation should be retried.

    Args:
        exception: The exception that occurred
        retry_count: Current retry attempt number
        max_retries: Maximum number of retries allowed

    Returns:
        True if should retry, False otherwise
    """
    if retry_count >= max_retries:
        return False

    # Retry on rate limits, timeouts, and server errors
    return (
        is_rate_limit_error(exception)
        or is_timeout_error(exception)
        or is_server_error(exception)
    )


def calculate_delay(
    retry_count: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    is_rate_limit: bool = False,
    rate_limit_pause: float = 65.0,
    add_jitter: bool = True
) -> float:
    """
    Calculate delay before next retry using exponential backoff with jitter.

    Args:
        retry_count: Current retry attempt number (0-indexed)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        is_rate_limit: True if error was a rate limit
        rate_limit_pause: Special pause duration for rate limits
        add_jitter: Add random jitter to prevent thundering herd (default: True)

    Returns:
        Delay in seconds before next retry
    """
    if is_rate_limit:
        # For rate limits, use special pause duration with small jitter
        jitter = random.uniform(0, 5) if add_jitter else 0
        return rate_limit_pause + jitter

    # Exponential backoff: initial_delay * (base ^ retry_count)
    delay = initial_delay * (exponential_base ** retry_count)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Add jitter (up to 25% of delay) to prevent thundering herd
    if add_jitter:
        jitter = random.uniform(0, delay * 0.25)
        delay += jitter

    return delay


def retry_with_backoff(
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    rate_limit_pause: float = 65.0
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay before first retry (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        rate_limit_pause: Special pause for rate limit errors (seconds)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, exponential_base=2.0)
        def call_api():
            return api.generate_content(prompt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retry_count = 0
            last_exception = None

            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if not should_retry(e, retry_count, max_retries):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                    # Calculate delay
                    is_rate_limit = is_rate_limit_error(e)
                    delay = calculate_delay(
                        retry_count,
                        initial_delay,
                        max_delay,
                        exponential_base,
                        is_rate_limit,
                        rate_limit_pause
                    )

                    # Log retry attempt
                    retry_count += 1
                    error_type = "rate limit" if is_rate_limit else "error"

                    if retry_count <= max_retries:
                        logger.warning(
                            f"{func.__name__} encountered {error_type}: {e}. "
                            f"Retry {retry_count}/{max_retries} after {delay:.1f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )

            # If we've exhausted retries, raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"{func.__name__} failed without exception")

        return wrapper
    return decorator


def retry_async_with_backoff(
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    rate_limit_pause: float = 65.0
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay before first retry (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        rate_limit_pause: Special pause for rate limit errors (seconds)

    Returns:
        Decorated async function with retry logic

    Example:
        @retry_async_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, exponential_base=2.0)
        async def call_api_async():
            return await api.generate_content_async(prompt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            import asyncio

            retry_count = 0
            last_exception = None

            while retry_count <= max_retries:
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if not should_retry(e, retry_count, max_retries):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                    # Calculate delay
                    is_rate_limit = is_rate_limit_error(e)
                    delay = calculate_delay(
                        retry_count,
                        initial_delay,
                        max_delay,
                        exponential_base,
                        is_rate_limit,
                        rate_limit_pause
                    )

                    # Log retry attempt
                    retry_count += 1
                    error_type = "rate limit" if is_rate_limit else "error"

                    if retry_count <= max_retries:
                        logger.warning(
                            f"{func.__name__} encountered {error_type}: {e}. "
                            f"Retry {retry_count}/{max_retries} after {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )

            # If we've exhausted retries, raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"{func.__name__} failed without exception")

        return wrapper
    return decorator
