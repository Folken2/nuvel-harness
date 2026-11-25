"""The model streams a run has open, so that its end closes them.

ADK's LiteLlm reads a streamed call with a bare ``async for`` and never closes it. When
a run is stopped, ADK unwinds at a ``yield`` inside that loop, the stream is dropped
unclosed, and its connection keeps reading the provider's answer to the end, which
the provider bills. So the driver opens a scope per run, the LiteLLM client puts every
stream it returns in it, and the driver closes what is still open once ADK is done.

The scope is a context variable: the tasks ADK starts for a run copy the driver's
context, so they all see the run's scope, and calls outside a run see none.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

_scope: contextvars.ContextVar[set[TrackedStream] | None] = contextvars.ContextVar(
    "model_streams", default=None
)


class TrackedStream:
    """The stream as ADK reads it, out of the scope once read to its end."""

    def __init__(self, stream: Any, scope: set[TrackedStream]) -> None:
        self._stream = stream
        self._scope = scope
        scope.add(self)

    def __aiter__(self) -> TrackedStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return await self._stream.__anext__()
        except StopAsyncIteration:
            self._scope.discard(self)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def aclose(self) -> None:
        self._scope.discard(self)
        # LiteLLM's wrapper closes the provider's HTTP response under it (since
        # 1.108; tests/test_run_stop_model_stream.py checks it on the wire).
        await self._stream.aclose()


def track(stream: Any) -> Any:
    """What the LiteLLM client returns: a streamed answer goes in the run's scope."""
    scope = _scope.get()
    if scope is None or not hasattr(stream, "__anext__") or not hasattr(stream, "aclose"):
        return stream
    return TrackedStream(stream, scope)


@asynccontextmanager
async def closing_model_streams() -> AsyncIterator[None]:
    """Around a run: on the way out, closes the streams it left open."""
    scope: set[TrackedStream] = set()
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)
        left_open = list(scope)
        for stream in left_open:
            try:
                await stream.aclose()
            except Exception:
                logger.warning("couldn't close a model stream", exc_info=True)
        if left_open:
            logger.info("closed %d model stream(s) the run left open", len(left_open))
