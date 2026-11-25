"""Streamed text cut into sentences, so speech can start before the reply ends."""

from __future__ import annotations

import re

# A sentence ends at . ! ? or a line break, followed by space. A number like 3.5 has none.
_END = re.compile(r"(?<=[.!?:;])[\"')\]]*\s+|\n+")
# Shorter pieces sound choppy when spoken one request at a time, so they wait for more.
MIN_CHARS = 24


class SentenceBuffer:
    """Takes text deltas; gives back each sentence once it's whole, with its spacing."""

    def __init__(self, min_chars: int = MIN_CHARS) -> None:
        self.min_chars = min_chars
        self._text = ""

    def push(self, delta: str) -> list[str]:
        self._text += delta
        out: list[str] = []
        start = 0
        for match in _END.finditer(self._text):
            if match.end() - start >= self.min_chars:
                out.append(self._text[start:match.end()])
                start = match.end()
        self._text = self._text[start:]
        return out

    def flush(self) -> str:
        rest, self._text = self._text, ""
        return rest
