"""Whether each plugin's connectors work right now, for the library's screens.

The same attach a turn makes, with nothing kept but the answer: a connector the
library shows as available is one a turn would have got. Every load of the library
asks, so the answer for a version is kept a short while (``answer_cache.py``) and only
a few connectors are tried at once, across every request: otherwise each page view
would send every approved plugin's credentials out again. A connector whose answer
won't change by asking again (it needs a sign-in) is remembered by its address for
longer (``lasting_ttl_seconds``), for every plugin that names it: the same few services
appear in many plugins.

It doesn't borrow a turn's toolsets. Those are one person's sessions, and a toolset
asked with no turn answers from ADK's per-invocation memory, so a kept one would say
a connector works long after it stopped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from ..plugin_source import Plugin, TurnPlugins
from ..user_tools.mcp_limits import McpLimits
from ..user_tools.mcp_toolsets import build_mcp_toolset, close_within
from ..user_tools.unavailable import UnavailableTools
from .answer_cache import AnswerCache
from .attach import AttachFailed, attach, some_names_unusable
from .missing import NOT_LOADED
from .mounts import Mount, connectors_of

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ConnectorStatus:
    name: str
    # None: it works. Otherwise why not, in the words a turn's model is given.
    problem: str | None


@dataclass(frozen=True)
class PluginStatus:
    name: str
    # None: it loaded. Otherwise why not; its connectors are then unknown.
    problem: str | None
    connectors: tuple[ConnectorStatus, ...]


@dataclass(frozen=True)
class _Tried:
    # None: it works. Otherwise why not.
    problem: str | None
    # Asking again won't change it.
    lasting: bool = False


def _address_key(mount: Mount) -> str:
    """One connector's address and headers, hashed: headers can carry a key."""
    server = mount.server
    seen = json.dumps([server.get("url"), server.get("transport"), server.get("headers")], sort_keys=True)
    return hashlib.sha256(seen.encode()).hexdigest()


class ConnectorCheck:
    def __init__(
        self,
        ttl_seconds: float,
        concurrency: int,
        *,
        limits: Callable[[], McpLimits],
        close_timeout_seconds: float,
        lasting_ttl_seconds: float = 0,
    ) -> None:
        """``limits`` is asked at each check, so a changed setting is seen. A server that
        won't hang up gets ``close_timeout_seconds``."""
        self._limits = limits
        self._close_timeout_seconds = close_timeout_seconds
        self._answers: AnswerCache[tuple[ConnectorStatus, ...]] = AnswerCache(ttl_seconds)
        self._addresses: AnswerCache[_Tried] = AnswerCache(
            0, ttl_of=lambda tried: lasting_ttl_seconds if tried.lasting else 0
        )
        self._slots = asyncio.Semaphore(concurrency)
        # Connections made, for the logs and the tests.
        self.attaches = 0

    async def check(self, plugins: TurnPlugins) -> list[PluginStatus]:
        loaded = await asyncio.gather(
            *(self._answers.get(plugin.content_hash, lambda p=plugin: self._connectors(p)) for plugin in plugins.loaded)
        )
        return [PluginStatus(p.name, None, c) for p, c in zip(plugins.loaded, loaded)] + [
            PluginStatus(name, NOT_LOADED, ()) for name in plugins.unavailable
        ]

    async def _connectors(self, plugin: Plugin) -> tuple[ConnectorStatus, ...]:
        # One plugin at a time loses nothing: a mount is "<plugin>_<connector>" and a
        # plugin's name is kebab-case (library/manifest.py), so no two plugins share one.
        mounts, missing = connectors_of([plugin])
        tried = await asyncio.gather(
            *(self._addresses.get(_address_key(mount), lambda m=mount: self._try(m)) for mount in mounts)
        )
        problems = {m.connector: m.reason for m in missing}
        problems.update((mount.connector, t.problem) for mount, t in zip(mounts, tried) if t.problem is not None)
        return tuple(ConnectorStatus(c.name, problems.get(c.name)) for c in plugin.components.connectors)

    async def _try(self, mount: Mount) -> _Tried:
        async with self._slots:
            self.attaches += 1
            logger.info("checking plugin connector: %s", mount.name)
            limits = self._limits()
            toolset = build_mcp_toolset(mount.server, UnavailableTools(), limits)
            try:
                attached = await attach(mount, toolset, None, limits)
            except AttachFailed as failed:
                return _Tried(failed.reason, failed.lasting)
            finally:
                await close_within(toolset, self._close_timeout_seconds, mount.name)
        if attached.left_out:
            return _Tried(some_names_unusable(attached.left_out))
        return _Tried(None)
