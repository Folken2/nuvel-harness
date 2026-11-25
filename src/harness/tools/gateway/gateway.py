"""One gateway per agent: the toolsets it defers, its own toolset and its plugin.

A product builds one with its threshold and pinned names, wraps each toolset whose servers
may be large in ``defer``, puts ``toolset`` last in the agent's tools and ``plugin()`` before
its approval gate. The three share what this step deferred.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from google.adk.tools.base_toolset import BaseToolset

from .deferred import DeferredTools
from .deferring import DeferringToolset
from .toolset import ToolGatewayToolset

if TYPE_CHECKING:
    from ...plugins.tool_gateway import ToolGatewayPlugin


class ToolGateway:
    def __init__(self, *, threshold_tokens: int, pinned: Iterable[str]) -> None:
        self.deferred = DeferredTools()
        self._threshold_tokens = threshold_tokens
        self._pinned = frozenset(pinned)
        self.toolset = ToolGatewayToolset(self.deferred)
        self._deferring: list[DeferringToolset] = []

    def defer(self, toolset: BaseToolset) -> DeferringToolset:
        deferring = DeferringToolset(
            toolset, self.deferred, threshold_tokens=self._threshold_tokens, pinned=self._pinned
        )
        self._deferring.append(deferring)
        return deferring

    def forget(self, invocation_id: str) -> None:
        """What a run that ended or paused left: its deferred tools and its listings."""
        self.deferred.forget(invocation_id)
        for deferring in self._deferring:
            deferring.forget(invocation_id)

    def plugin(self) -> ToolGatewayPlugin:
        # Imported here: the plugin imports this package (its answers for a tool not loaded),
        # so importing it at the top would make a cycle for whoever imports the plugin first.
        from ...plugins.tool_gateway import ToolGatewayPlugin

        return ToolGatewayPlugin(self.deferred, self.forget)
