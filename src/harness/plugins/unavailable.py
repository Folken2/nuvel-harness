"""Tells the web app which connected apps couldn't be loaded for a turn.

When an MCP server or a hosted toolkit's session is down, the turn goes on
without those tools. The toolset records what was skipped, and this plugin
puts the names on the next event as ``custom_metadata["<namespace>_unavailable_tools"]``
so the web app can warn the user.
"""

from __future__ import annotations

from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.plugins.base_plugin import BasePlugin

from .. import wire_names
from ..tools.user_tools.unavailable import UnavailableTools


class UnavailableToolsPlugin(BasePlugin):
    def __init__(self, registry: UnavailableTools) -> None:
        super().__init__(name="unavailable_tools")
        self._registry = registry

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        names = self._registry.take(invocation_context.invocation_id)
        if names:
            event.custom_metadata = {
                **(event.custom_metadata or {}),
                wire_names.UNAVAILABLE_TOOLS: names,
            }
        # In place, returning nothing: a returned event would end the chain here and
        # hide this event from the plugins after this one.
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        self._registry.forget(invocation_context.invocation_id)
