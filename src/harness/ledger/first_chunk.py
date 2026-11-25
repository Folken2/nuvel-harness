"""A streamed answer's first chunk, read while its request can still fail.

Some of LiteLLM's streams are lazy: the direct Gemini route sends nothing until the
stream is first read, so a rate limit, a too-long prompt or a refused key raises on the
first chunk, after the client has already returned. Reading that chunk inside the
attempt makes it fail as any refused request does: it settles in the ledger and the
retry loop (``routing/client.py``) sees it. Only the first chunk is read early, so
nothing the person could have seen is ever sent again.
"""

from __future__ import annotations

from typing import Any

_NOTHING = object()


class PrimedStream:
    """The stream with its first chunk already in hand; reads on from there."""

    def __init__(self, stream: Any, first: Any) -> None:
        self._stream = stream
        self._first = first

    def __aiter__(self) -> "PrimedStream":
        return self

    async def __anext__(self) -> Any:
        if self._first is not _NOTHING:
            chunk, self._first = self._first, _NOTHING
            return chunk
        return await self._stream.__anext__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def aclose(self) -> None:
        await self._stream.aclose()


async def primed(stream: Any) -> Any:
    """``stream`` with its first chunk read; whatever that read raises, raised here."""
    if not hasattr(stream, "__anext__"):
        return stream
    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        return stream
    except BaseException:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass
        raise
    return PrimedStream(stream, first)
