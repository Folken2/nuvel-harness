"""16-bit mono PCM: its rate, its loudness, and the WAV header a clip needs."""

from __future__ import annotations

import array
import math
import struct
import sys
from typing import Optional

__all__ = ["duration_seconds", "rate_of"]

SAMPLE_BYTES = 2


def rate_of(mime_type: Optional[str], default: int) -> int:
    """The `rate=` of a mime type such as `audio/pcm;rate=16000`."""
    for token in (mime_type or "").split(";"):
        token = token.strip()
        if token.startswith("rate="):
            try:
                return int(token[len("rate="):])
            except ValueError:
                return default
    return default


def rms(pcm: bytes) -> float:
    """Loudness from 0 to 1."""
    usable = len(pcm) - len(pcm) % SAMPLE_BYTES
    if usable == 0:
        return 0.0
    samples = array.array("h", pcm[:usable])
    if sys.byteorder == "big":
        samples.byteswap()
    total = sum(sample * sample for sample in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def duration_seconds(pcm: bytes, rate: int) -> float:
    return len(pcm) / SAMPLE_BYTES / rate


def wav(pcm: bytes, rate: int) -> bytes:
    """The PCM in a WAV file, which every transcription model takes."""
    usable = pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES]
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(usable), b"WAVE", b"fmt ", 16, 1, 1,
        rate, rate * SAMPLE_BYTES, SAMPLE_BYTES, 16, b"data", len(usable),
    )
    return header + usable
