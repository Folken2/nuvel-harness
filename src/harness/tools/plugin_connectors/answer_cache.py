"""Answers kept for a short while by key, and one computation per key at a time.

Two callers asking the same thing at once share one computation; a caller within the
window gets the kept answer. A computation that raises is not kept. The window can
depend on the answer (``ttl_of``): 0 keeps nothing.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class AnswerCache(Generic[T]):
    def __init__(
        self,
        ttl_seconds: float,
        limit: int = 256,
        clock: Callable[[], float] = time.monotonic,
        ttl_of: Callable[[T], float] | None = None,
    ) -> None:
        self._ttl_of: Callable[[T], float] = ttl_of or (lambda _value: ttl_seconds)
        self._limit = limit
        self._clock = clock
        self._kept: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._running: dict[str, asyncio.Task[T]] = {}

    async def get(self, key: str, compute: Callable[[], Awaitable[T]]) -> T:
        kept = self._kept.get(key)
        if kept is not None and kept[0] > self._clock():
            return kept[1]
        running = self._running.get(key)
        if running is None:
            running = asyncio.ensure_future(self._compute(key, compute))
            self._running[key] = running
        # Shielded: a caller that leaves doesn't cancel the others' answer.
        return await asyncio.shield(running)

    async def _compute(self, key: str, compute: Callable[[], Awaitable[T]]) -> T:
        try:
            value = await compute()
        finally:
            self._running.pop(key, None)
        ttl = self._ttl_of(value)
        if ttl > 0:
            self._kept[key] = (self._clock() + ttl, value)
            self._kept.move_to_end(key)
            while len(self._kept) > self._limit:
                self._kept.popitem(last=False)
        return value
