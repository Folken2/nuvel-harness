"""Text to speech over a provider that speaks one text per call, as ADK's LiveEgress.

The provider is what the caller passes (a sibling module binds this service's): the
``speak`` and ``price_per_char`` callables, and the ``mime_type`` of the audio it sends.

Text deltas are cut into sentences (``sentences.py``) and each one is spoken as soon as
it's whole, so audio starts before the reply ends. The next sentence's call starts while
the one before it is still arriving. ``cancel`` (a barge-in) stops at the next chunk.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Optional

from google.adk.live import AgentSpokenOutput, AudioChunk, EgressEvent, LiveEgress
from google.genai import types

from .sentences import SentenceBuffer
from .speakable import speakable
from .usage import AudioUsage, UsageSink

logger = logging.getLogger(__name__)

# One sentence's audio held while the one before it plays out: bounded, so a long
# reply doesn't sit in memory.
_AHEAD = 1

# Called as speak(words, key=..., model=..., voice=...): the audio as it arrives.
Speak = Callable[..., AsyncIterator[bytes]]
# Called as price_per_char(model, key=...): None when the price isn't known.
Price = Callable[..., Awaitable[Optional[float]]]


class SentenceTTS(LiveEgress):
    def __init__(
        self,
        *,
        key: str,
        model: str,
        voice: str,
        speak: Speak,
        price_per_char: Price,
        mime_type: str,
        on_usage: Optional[UsageSink] = None,
    ) -> None:
        self.key = key
        self.model = model
        self.voice = voice
        self.on_usage = on_usage
        self._speak = speak
        self._price = price_per_char
        self.mime_type = mime_type

    async def _record(self, words: str) -> None:
        if not self.on_usage:
            return
        price = await self._price(self.model, key=self.key)
        if price is None:
            logger.warning("no speech price for %s; its cost isn't recorded", self.model)
        cost = (price or 0.0) * len(words)
        self.on_usage(AudioUsage("text_to_speech", self.model, cost, float(len(words))))

    async def _fetch(self, words: str, out: asyncio.Queue[Optional[bytes]]) -> None:
        try:
            async for chunk in self._speak(words, key=self.key, model=self.model, voice=self.voice):
                await out.put(chunk)
        finally:
            await out.put(None)

    async def __call__(
        self, text: AsyncIterator[str], *, cancel: asyncio.Event
    ) -> AsyncGenerator[EgressEvent, None]:
        sentences: asyncio.Queue[Optional[str]] = asyncio.Queue()
        buffer = SentenceBuffer()

        async def read_text() -> None:
            try:
                async for delta in text:
                    for sentence in buffer.push(delta):
                        await sentences.put(sentence)
                rest = buffer.flush()
                if rest.strip():
                    await sentences.put(rest)
            finally:
                await sentences.put(None)

        reader = asyncio.create_task(read_text())
        # (sentence as written, its audio as it arrives, the call making it)
        pending: list[tuple[str, asyncio.Queue[Optional[bytes]], asyncio.Task[None]]] = []
        try:
            done_reading = False
            while not cancel.is_set():
                # Start calls up to one sentence ahead of the one playing.
                while not done_reading and len(pending) <= _AHEAD:
                    if pending and sentences.empty() and not reader.done():
                        break
                    sentence = await sentences.get()
                    if sentence is None:
                        done_reading = True
                        break
                    words = speakable(sentence)
                    if not words:
                        continue
                    await self._record(words)
                    audio: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
                    pending.append((sentence, audio, asyncio.create_task(self._fetch(words, audio))))
                if not pending:
                    if done_reading:
                        return
                    continue
                sentence, audio, call = pending.pop(0)
                while not cancel.is_set():
                    chunk = await audio.get()
                    if chunk is None:
                        break
                    yield AudioChunk(blob=types.Blob(data=chunk, mime_type=self.mime_type))
                if cancel.is_set():
                    return
                await call  # Raises the call's error, if it had one.
                yield AgentSpokenOutput(text=sentence)
        finally:
            reader.cancel()
            for _, _, call in pending:
                call.cancel()
            for task in [reader, *(call for _, _, call in pending)]:
                with contextlib.suppress(BaseException):
                    await task
