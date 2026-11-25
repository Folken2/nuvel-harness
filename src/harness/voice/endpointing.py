"""Where an utterance starts and ends in a stream of PCM, by its loudness.

The transcription API takes a whole clip, so the ingress finds the clips itself.
Plain energy against a noise floor that follows the room: enough for a microphone with
the browser's echo cancellation on, and it costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

FRAME_MS = 20


class Change(Enum):
    NONE = "none"
    STARTED = "started"
    ENDED = "ended"


@dataclass
class EndpointSettings:
    # Speech is this many times louder than the room, and at least `min_level`.
    ratio: float = 3.0
    min_level: float = 0.015
    # Speech this long starts an utterance, so a click doesn't.
    start_ms: int = 120
    # Quiet this long ends one.
    end_ms: int = 700


class Endpointer:
    """Fed one frame's loudness at a time; says when speech starts and ends."""

    def __init__(self, settings: EndpointSettings | None = None) -> None:
        self.settings = settings or EndpointSettings()
        self.floor = self.settings.min_level / self.settings.ratio
        self.speaking = False
        self._loud_ms = 0
        self._quiet_ms = 0

    def _is_loud(self, level: float) -> bool:
        return level >= max(self.settings.min_level, self.floor * self.settings.ratio)

    def feed(self, level: float, frame_ms: int = FRAME_MS) -> Change:
        loud = self._is_loud(level)
        if not loud:
            # The floor follows the room only while nobody is speaking over it.
            self.floor = self.floor * 0.95 + level * 0.05
        if not self.speaking:
            self._loud_ms = self._loud_ms + frame_ms if loud else 0
            if self._loud_ms >= self.settings.start_ms:
                self.speaking = True
                self._quiet_ms = 0
                return Change.STARTED
            return Change.NONE
        self._quiet_ms = 0 if loud else self._quiet_ms + frame_ms
        if self._quiet_ms >= self.settings.end_ms:
            self.speaking = False
            self._loud_ms = 0
            return Change.ENDED
        return Change.NONE
