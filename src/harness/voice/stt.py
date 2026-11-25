"""Speech to text over a provider that transcribes whole clips, as ADK's LiveIngress.

The provider is the ``transcribe`` callable the caller passes (a sibling module binds
this service's). It takes one clip at a time, so this finds the utterances itself
(``endpointing.py``) and sends each one when it ends. While a person speaks it sends
the clip so far every ``partial_every_seconds``, one call at a time, for captions: at
the cheapest models that is cents per hour. With ``endpoint=None`` the caller marks the
ends instead, as CascadeLive does on ``ActivityEnd``: each segment is one utterance.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Optional, Protocol

from google.adk.live import (
    IngressEvent,
    LiveIngress,
    PartialTranscript,
    UserSpeechStarted,
    UserTurnFinished,
)
from google.genai import types

from . import pcm
from .endpointing import FRAME_MS, Change, Endpointer, EndpointSettings
from .usage import AudioUsage, UsageSink

logger = logging.getLogger(__name__)

# What a browser microphone sends once resampled; a blob's mime type can say otherwise.
DEFAULT_RATE = 16_000
# Audio kept from before speech was detected, so the first word isn't cut: detecting
# speech takes EndpointSettings.start_ms of it, and a little more is kept before that.
PREROLL_MS = 500



class Heard(Protocol):
    """What a transcription call answers with."""

    @property
    def text(self) -> str: ...

    @property
    def cost_usd(self) -> float: ...

    @property
    def seconds(self) -> float: ...


# Called as transcribe(wav, key=..., model=..., language=...).
Transcribe = Callable[..., Awaitable[Heard]]


class ClipSTT(LiveIngress):
    def __init__(
        self,
        *,
        key: str,
        model: str,
        transcribe: Transcribe,
        language: Optional[str] = None,
        endpoint: Optional[EndpointSettings] = None,
        partial_every_seconds: Optional[float] = 1.2,
        on_usage: Optional[UsageSink] = None,
    ) -> None:
        self.key = key
        self.model = model
        self.language = language
        self.endpoint = endpoint
        self.partial_every_seconds = partial_every_seconds
        self.on_usage = on_usage
        self._transcribe = transcribe

    async def transcribe_clip(self, clip: bytes, rate: int) -> str:
        """One clip's words, its cost reported."""
        result = await self._transcribe(
            pcm.wav(clip, rate), key=self.key, model=self.model, language=self.language
        )
        if self.on_usage:
            self.on_usage(
                AudioUsage("speech_to_text", self.model, result.cost_usd, result.seconds)
            )
        return result.text

    async def __call__(
        self, audio: AsyncIterator[types.Blob]
    ) -> AsyncGenerator[IngressEvent, None]:
        rate = DEFAULT_RATE
        endpointer = Endpointer(self.endpoint) if self.endpoint else None
        preroll = bytearray()
        utterance = bytearray()
        speaking = False
        partial: Optional[asyncio.Task[str]] = None
        partial_at = 0.0

        def partial_due() -> bool:
            if self.partial_every_seconds is None or partial is not None:
                return False
            heard = pcm.duration_seconds(bytes(utterance), rate)
            return heard - partial_at >= self.partial_every_seconds

        async def finish() -> AsyncGenerator[IngressEvent, None]:
            nonlocal partial, speaking, partial_at
            if partial is not None:
                partial.cancel()
                with contextlib.suppress(BaseException):
                    await partial
                partial = None
            clip = bytes(utterance)
            utterance.clear()
            speaking = False
            partial_at = 0.0
            if pcm.duration_seconds(clip, rate) < 0.2:
                return
            text = await self.transcribe_clip(clip, rate)
            if text:
                yield UserTurnFinished(text=text)

        try:
            async for blob in audio:
                data = blob.data or b""
                rate = pcm.rate_of(blob.mime_type, rate)
                frame_bytes = rate * FRAME_MS // 1000 * pcm.SAMPLE_BYTES
                for start in range(0, len(data), frame_bytes):
                    frame = data[start:start + frame_bytes]
                    if endpointer is None:
                        if not speaking:
                            speaking = True
                            yield UserSpeechStarted()
                        utterance.extend(frame)
                        continue
                    change = endpointer.feed(pcm.rms(frame))
                    if change is Change.STARTED:
                        speaking = True
                        utterance.extend(preroll)
                        preroll.clear()
                        yield UserSpeechStarted()
                    if speaking:
                        utterance.extend(frame)
                    else:
                        preroll.extend(frame)
                        keep = rate * PREROLL_MS // 1000 * pcm.SAMPLE_BYTES
                        del preroll[: max(0, len(preroll) - keep)]
                    if change is Change.ENDED:
                        async for event in finish():
                            yield event
                if partial is not None and partial.done():
                    done, partial = partial, None
                    if not done.cancelled() and done.exception() is None and done.result():
                        yield PartialTranscript(text=done.result())
                    elif not done.cancelled() and done.exception() is not None:
                        logger.warning("partial transcription failed: %s", done.exception())
                if speaking and partial_due():
                    partial_at = pcm.duration_seconds(bytes(utterance), rate)
                    partial = asyncio.create_task(self.transcribe_clip(bytes(utterance), rate))
            if speaking:
                async for event in finish():
                    yield event
        finally:
            if partial is not None:
                partial.cancel()
