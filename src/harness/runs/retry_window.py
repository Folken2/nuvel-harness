"""What a run may still spend waiting on retries, from the run's limits to the client.

The run's time ceiling is checked before each model call (``runs/ceiling.py``), so
waits inside one call would slip past it. The limits plugin hands each call its run's
window through a context variable, like the call's key (``routing/turn_route.py``):
the moment the ceiling ends, and the retry wait the run has left. The client spends
from it. A call outside a run (a title, a test) has none and keeps to its own budget.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetryWindow:
    # time.monotonic() when the run's time ceiling ends.
    deadline: float
    # Retry waiting the run has left, across all its calls.
    wait_left: float


_window: contextvars.ContextVar[Optional[RetryWindow]] = contextvars.ContextVar(
    "retry_window", default=None
)


def use_window(window: Optional[RetryWindow]) -> None:
    _window.set(window)


def current_window() -> Optional[RetryWindow]:
    return _window.get()


def wait_left(call_wait_left: float, window: Optional[RetryWindow], now: Optional[float] = None) -> float:
    """The longest wait the next retry may take: the call's own, the run's, and the
    time to its ceiling, whichever is least."""
    if window is None:
        return call_wait_left
    now = time.monotonic() if now is None else now
    return max(min(call_wait_left, window.wait_left, window.deadline - now), 0.0)
