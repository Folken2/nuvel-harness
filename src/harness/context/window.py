"""How full a model's context may get before history is rolled up, from its window.

One fixed number fits no model: a small window overflows before it is reached, and a
large one is rolled up long before it needs to be, losing detail for nothing. So the
threshold is a share of the window of the model the run is on, never above a ceiling
that keeps a very large window from making every call expensive. On a small window the
share alone sits close to what every call carries anyway (a 32k model at half is 16k,
about a plain prompt plus the carried summary), so it would roll up on most calls: a
floor lifts it, never past ``FLOOR_SHARE`` of the window, which keeps room for the reply.
A model whose window isn't known keeps the fixed number.

Pure. The share, the ceiling and the fixed number are the product's settings; the window
comes with the turn (``plugins/context_budget.py`` is handed how to read it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# The most of a window the floor may lift the threshold to.
FLOOR_SHARE = 0.75


@dataclass(frozen=True)
class WindowSettings:
    """History is rolled up once a prompt reaches ``fraction`` of the model's window,
    and at ``ceiling`` tokens at the latest; never below ``floor`` tokens while that is
    within ``FLOOR_SHARE`` of the window. A fraction of 0 keeps the fixed number."""

    fraction: float
    ceiling: int
    floor: int = 0


def window_of(value: object) -> Optional[int]:
    """A window as it arrives with a turn: a positive whole number of tokens, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= 1 else None


def threshold_for(window: Optional[int], settings: WindowSettings, fixed: int) -> int:
    """The prompt size, in tokens, at which this model's history is rolled up."""
    if window is None or settings.fraction <= 0:
        return fixed
    share = int(window * settings.fraction)
    lifted = max(share, min(settings.floor, int(window * FLOOR_SHARE)))
    return max(1, min(lifted, settings.ceiling))
