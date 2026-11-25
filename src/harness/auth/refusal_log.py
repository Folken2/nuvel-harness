"""How many refusals of callers without a valid token reach the log.

Anyone can send those, so not all of them. Each line names only a method and a route
word (middleware.py), so nothing a caller sends is written as it came. What's left to
bound is the number: past a few lines a minute they are counted, and the first line
after the minute says how many went unlogged. Refusals of a valid token aren't capped
(only the web app can sign one), so a flood of these can't hide them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

LINES_PER_WINDOW = 20
WINDOW_SECONDS = 60.0


class RefusalLog:
    def __init__(
        self,
        logger: logging.Logger,
        lines_per_window: int = LINES_PER_WINDOW,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logger = logger
        self._lines_per_window = lines_per_window
        self._window_seconds = window_seconds
        self._clock = clock
        self._window_start = clock()
        self._logged = 0
        self._dropped = 0

    def refused(self, message: str, *args: object) -> None:
        now = self._clock()
        if now - self._window_start >= self._window_seconds:
            if self._dropped:
                self._logger.warning(
                    "%d more refused requests went unlogged in the minute before", self._dropped
                )
            self._window_start, self._logged, self._dropped = now, 0, 0
        if self._logged < self._lines_per_window:
            self._logged += 1
            self._logger.warning(message, *args)
        else:
            self._dropped += 1
