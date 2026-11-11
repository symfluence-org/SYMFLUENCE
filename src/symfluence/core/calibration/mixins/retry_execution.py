# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Retry Execution Mixin

Provides retry logic with exponential backoff for model evaluations.
Handles transient failures like stale file handles and I/O errors.
"""
from __future__ import annotations

import logging
import random
import signal
import time
from functools import wraps
from typing import Any, Callable, Optional

from symfluence.core.exceptions import RetryExhaustedError
from symfluence.core.mixins import ConfigMixin

logger = logging.getLogger(__name__)


# Transient errors that warrant retry
TRANSIENT_ERRORS = (
    'stale file handle',
    'resource temporarily unavailable',
    'no such file or directory',
    'permission denied',
    'connection refused',
    'broken pipe',
    'network is unreachable',
)


def is_transient_error(error: Exception) -> bool:
    """
    Check if an error is likely transient and worth retrying.

    Args:
        error: The exception to check

    Returns:
        True if the error is likely transient
    """
    error_str = str(error).lower()
    return any(te in error_str for te in TRANSIENT_ERRORS)


class RetryExecutionMixin(ConfigMixin):
    """
    Mixin class providing retry logic with exponential backoff.

    Requires the following attributes on the class using this mixin:
    - self.config: Dict[str, Any]
    - self.logger: logging.Logger

    Provides:
    - Retry logic with exponential backoff
    - Signal handling for clean termination
    - Staggered worker starts
    """

    # =========================================================================
    # Configuration
    # =========================================================================

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        return self._get_config_value(
            lambda: self.config.optimization.worker_max_retries,
            default=3
        )

    @property
    def base_delay(self) -> float:
        """Base delay for exponential backoff (seconds)."""
        return self._get_config_value(
            lambda: self.config.optimization.worker_base_delay,
            default=0.5
        )

    @property
    def max_delay(self) -> float:
        """Maximum delay between retries (seconds)."""
        return self._get_config_value(
            lambda: self.config.optimization.worker_max_delay,
            default=30.0
        )

    @property
    def jitter_factor(self) -> float:
        """Jitter factor for randomizing delays (0-1)."""
        return self._get_config_value(
            lambda: self.config.optimization.worker_jitter,
            default=0.1
        )

    # =========================================================================
    # Retry logic
    # =========================================================================

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        retry_on: Optional[tuple] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic and exponential backoff.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            max_retries: Maximum retries (default: self.max_retries)
            retry_on: Tuple of exception types to retry on
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            The last exception if all retries fail
        """
        if max_retries is None:
            max_retries = self.max_retries

        if retry_on is None:
            retry_on = (Exception,)

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)

            except retry_on as e:
                last_exception = e

                if attempt >= max_retries:
                    self.logger.error(
                        f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}"
                    )
                    raise

                # Check if error is transient
                if not is_transient_error(e):
                    self.logger.warning(
                        f"Non-transient error, not retrying: {e}"
                    )
                    raise

                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_delay(attempt)

                self.logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

        if last_exception is not None:
            raise last_exception
        raise RetryExhaustedError(f"Function {func.__name__} failed: retry loop completed without success or captured error")

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.base_delay * (2 ** attempt)

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter
        jitter = delay * self.jitter_factor * random.random()
        delay += jitter

        return delay

    def retry_decorator(
        self,
        max_retries: Optional[int] = None,
        retry_on: Optional[tuple] = None
    ) -> Callable:
        """
        Create a retry decorator.

        Args:
            max_retries: Maximum retries
            retry_on: Exception types to retry on

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_retry(
                    func, *args,
                    max_retries=max_retries,
                    retry_on=retry_on,
                    **kwargs
                )
            return wrapper
        return decorator

    # =========================================================================
    # Signal handling
    # =========================================================================

    def setup_signal_handlers(self) -> None:
        """
        Setup signal handlers for clean termination.

        Handles SIGTERM and SIGINT for graceful shutdown.
        """
        self._shutdown_requested = False

        def signal_handler(signum, frame):
            self._shutdown_requested = True
            self.logger.info(f"Received signal {signum}, requesting shutdown...")

        # Only set handlers in main thread
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Signal handling only works in main thread
            pass

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return getattr(self, '_shutdown_requested', False)

    def check_shutdown(self) -> None:
        """
        Check if shutdown was requested and raise if so.

        Raises:
            KeyboardInterrupt: If shutdown was requested
        """
        if self.shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested")

    # =========================================================================
    # Worker start staggering
    # =========================================================================

    def add_initial_delay(self, max_delay: float = 1.0) -> None:
        """
        Add a random initial delay to stagger worker starts.

        This helps prevent simultaneous file access conflicts
        when multiple workers start at the same time.

        Args:
            max_delay: Maximum delay in seconds
        """
        delay = random.uniform(0, max_delay)
        time.sleep(delay)


# ============================================================================
# Standalone utility functions (for use in worker scripts)
# ============================================================================

def retry_on_io_error(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 0.5
) -> Any:
    """
    Standalone retry function for I/O operations.

    Args:
        func: Function to execute (takes no arguments)
        max_retries: Maximum number of retries
        base_delay: Base delay for backoff

    Returns:
        Result from func
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except (IOError, OSError, PermissionError) as e:
            last_exception = e

            if attempt >= max_retries:
                raise

            if not is_transient_error(e):
                raise

            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
            time.sleep(delay)

    if last_exception is not None:
        raise last_exception
    raise RetryExhaustedError(f"Function {func.__name__} failed: retry loop completed without success or captured error")


def with_staggered_start(func: Callable, max_delay: float = 0.5) -> Callable:
    """
    Decorator to add random initial delay to a function.

    Args:
        func: Function to wrap
        max_delay: Maximum delay in seconds

    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        time.sleep(random.uniform(0, max_delay))
        return func(*args, **kwargs)
    return wrapper
