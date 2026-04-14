import time
import logging
from functools import wraps
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
)

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
    ConnectionError,
    TimeoutError,
)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
):
    """
    Decorator that retries a function on transient Gemini API failures
    using exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Cap on the delay between retries.
        backoff_factor: Multiplier applied to the delay after each retry.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            "Function '%s' failed after %d attempts: %s",
                            func.__name__,
                            max_retries + 1,
                            str(e),
                        )
                        raise

                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(
                        "Function '%s' attempt %d/%d failed (%s: %s). "
                        "Retrying in %.1fs...",
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        type(e).__name__,
                        str(e),
                        delay,
                    )
                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator