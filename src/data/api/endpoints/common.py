"""Common utilities for ESPN API endpoints."""

import contextlib
from typing import AsyncContextManager, Optional, TypeVar, cast
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_async_context(manager: Optional[T], should_close: bool) -> AsyncContextManager[T]:
    """
    Get an appropriate async context manager.

    Args:
        manager: The context manager to use if should_close is False
        should_close: Whether to close the manager when exiting the context

    Returns:
        An async context manager
    """
    # If should_close is True, return the manager as is for context management
    if should_close:
        return cast(AsyncContextManager[T], manager)

    # Otherwise, return a nullcontext that doesn't close the manager
    # Using an async compatible version of nullcontext
    return contextlib.AsyncExitStack()
