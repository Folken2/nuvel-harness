"""The turn's plugin connectors: the MCP servers the person's installed plugins name.

Each connector is mounted by itself, so one that is broken costs only its own tools:
the plugin's skills, its other connectors and the rest of the turn go on. What's
missing is told twice. The model gets it in its instruction (``missing.py``, closed
by the words this toolset is given), so it can say which connector a task needed. The person gets the notice a skipped app
gives (``UnavailableTools``), for a connector that should have worked: one this
service can never run (a local command, a missing value) is in the library instead,
not in every turn's notice.

A connector's MCP session is one person's. Its headers are the plugin's, the same for
everyone who has it, but a server may keep state per session (a chosen workspace, a
cursor), and that must not carry from one person's turn to another's.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext

from ..plugin_source import PluginSource
from ..user_tools.mcp_limits import McpLimits
from ..user_tools.mcp_toolsets import McpToolsetCache
from ..user_tools.unavailable import UnavailableTools
from .attach import AttachFailed, attach, no_answer, some_names_unusable
from .missing import NOT_LOADED, Missing, notice
from .mounts import Mount, connectors_of

logger = logging.getLogger(__name__)

# Turns in flight at once, each holding what its plugins bring.
_TURNS_KEPT = 64


@dataclass(frozen=True)
class _Turn:
    mounts: tuple[Mount, ...]
    # Known before anything is tried: plugins that didn't load, connectors that can't run here.
    missing: tuple[Missing, ...]


class PluginConnectorToolset(BaseToolset):
    def __init__(
        self,
        unavailable: UnavailableTools,
        loader: PluginSource,
        limits: McpLimits,
        *,
        closing: str,
    ) -> None:
        super().__init__()
        # The paragraph after the list of what's missing, in the model's instruction.
        self._closing = closing
        self._loader = loader
        self._unavailable = unavailable
        # One entry per person and connector, so a service gives this cache a limit of
        # its own, above the person's apps'.
        self._limits = limits
        self._mcp_cache = McpToolsetCache(unavailable, self._limits)
        self._turns: OrderedDict[tuple[str, tuple[str, ...]], _Turn] = OrderedDict()
        # What the latest model step of each run goes without, for its instruction.
        self._missing: OrderedDict[str, tuple[Missing, ...]] = OrderedDict()

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        if readonly_context is None:
            return []
        invocation_id = readonly_context.invocation_id
        turn = await self._for_turn(readonly_context)
        if turn is None:
            self._missing.pop(invocation_id, None)
            return []

        owner = readonly_context.user_id

        async def load(mount: Mount) -> tuple[list[BaseTool], Missing | None]:
            if self._unavailable.is_hung(invocation_id, mount.name):
                return [], Missing(mount.plugin, mount.connector, no_answer(self._limits))
            try:
                attached = await attach(
                    mount,
                    self._mcp_cache.toolset_for(mount.server, owner),
                    readonly_context,
                    self._limits,
                )
            except AttachFailed as failed:
                if failed.hung:
                    self._unavailable.add_hung(invocation_id, mount.name)
                else:
                    self._unavailable.add(invocation_id, mount.name)
                return [], Missing(mount.plugin, mount.connector, failed.reason)
            if attached.left_out:
                # It came, short of some tools: the model hears which connector, so it
                # doesn't take the rest for all there is.
                return attached.tools, Missing(mount.plugin, mount.connector, some_names_unusable(attached.left_out))
            return attached.tools, None

        # In parallel: a connector that is down costs its timeout once, not per connector.
        loaded = await asyncio.gather(*(load(mount) for mount in turn.mounts))
        failed = tuple(missing for _, missing in loaded if missing is not None)
        self._remember(invocation_id, turn.missing + failed)
        return [tool for tools, _ in loaded for tool in tools]

    async def process_llm_request(self, *, tool_context: ToolContext, llm_request: LlmRequest) -> None:
        # ADK asks for the tools first, so this step's list is already here.
        missing = self._missing.get(tool_context.invocation_id)
        if missing:
            llm_request.append_instructions([notice(missing, self._closing)])

    async def _for_turn(self, context: ReadonlyContext) -> _Turn | None:
        state = context.state
        key = (context.invocation_id, self._loader.pinned_hashes(state))
        if not key[1]:
            return None
        if key in self._turns:
            self._turns.move_to_end(key)
            return self._turns[key]
        try:
            plugins = await self._loader.for_turn(state)
        except Exception:  # noqa: BLE001 - the turn goes on without its plugins' connectors
            logger.exception("the turn's plugins couldn't be read for their connectors")
            return None
        mounts, cannot_run = connectors_of(plugins.loaded)
        not_loaded = [Missing(name, None, NOT_LOADED) for name in plugins.unavailable]
        for name in plugins.unavailable:
            self._unavailable.add(context.invocation_id, name)
        turn = _Turn(tuple(mounts), tuple(not_loaded + cannot_run))
        self._turns[key] = turn
        while len(self._turns) > _TURNS_KEPT:
            self._turns.popitem(last=False)
        return turn

    def _remember(self, invocation_id: str, missing: tuple[Missing, ...]) -> None:
        self._missing[invocation_id] = missing
        self._missing.move_to_end(invocation_id)
        while len(self._missing) > _TURNS_KEPT:
            self._missing.popitem(last=False)

    async def close(self) -> None:
        await self._mcp_cache.close()
        self._turns.clear()
        self._missing.clear()
