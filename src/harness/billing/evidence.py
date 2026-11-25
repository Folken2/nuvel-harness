"""What of a provider's raw usage the ledger may keep (usage-and-analytics.md UG-9).

The ledger never stores prompts, replies, headers or credentials. A provider's
usage object can carry any of those next to its counters, so only numeric
billing counters under known key names and a few labels survive.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

MAX_VALUES = 500
MAX_DEPTH = 6
MAX_ARRAY = 40
MAX_KEY = 80

KEY_WORDS = (
    "token", "usage", "cache", "cost", "duration", "second", "image", "text",
    "audio", "count", "total", "input", "output", "reasoning", "ephemeral",
    "request", "megapixel", "modality", "currency", "width", "height", "raw",
)
LABELS = frozenset({"TEXT", "IMAGE", "AUDIO", "VIDEO", "USD", "EUR", "GBP"})

_PROVIDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+~-]{0,255}")


def _key_allowed(key: object) -> bool:
    if not isinstance(key, str) or not key or len(key) > MAX_KEY:
        return False
    lowered = key.lower()
    return any(word in lowered for word in KEY_WORDS)


class _Budget:
    def __init__(self) -> None:
        self.left = MAX_VALUES

    def take(self) -> bool:
        if self.left <= 0:
            return False
        self.left -= 1
        return True


def _leaf(value: object) -> Optional[object]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, str) and value in LABELS:
        return value
    return None


def _clean(value: object, depth: int, budget: _Budget) -> Optional[object]:
    if isinstance(value, dict):
        if depth >= MAX_DEPTH:
            return None
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not _key_allowed(key):
                continue
            cleaned = _clean(item, depth + 1, budget)
            if cleaned is not None:
                out[key] = cleaned
        return out or None
    if isinstance(value, (list, tuple)):
        if depth >= MAX_DEPTH:
            return None
        items = [_clean(item, depth + 1, budget) for item in list(value)[:MAX_ARRAY]]
        kept = [item for item in items if item is not None]
        return kept or None
    leaf = _leaf(value)
    if leaf is None or not budget.take():
        return None
    return leaf


def sanitise_provider_usage(raw: object) -> dict[str, Any]:
    """Only the counters and labels UG-9 allows; ``{}`` when nothing is left."""
    if not isinstance(raw, dict):
        return {}
    cleaned = _clean(raw, 0, _Budget())
    return cleaned if isinstance(cleaned, dict) else {}


def provider_id(value: object) -> Optional[str]:
    """A provider response or request id, or None when it isn't a safe one."""
    if not isinstance(value, str) or not _PROVIDER_ID.fullmatch(value):
        return None
    return value
