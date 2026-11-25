"""Answers a call to a deferred tool that isn't loaded (tools/gateway/not_loaded.py).

It returns a value from the before-tool callback, so it goes before the approval gate: a
call that can't run is never put to the person.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from ..tools.gateway.deferred import DeferredTools
from ..tools.gateway.not_loaded import not_loaded_answer


class ToolGatewayPlugin(BasePlugin):
    def __init__(self, deferred: DeferredTools, forget: Callable[[str], None]) -> None:
        super().__init__(name="tool_gateway")
        self._deferred = deferred
        # Drops what the gateway kept for a run once it ends or pauses.
        self._forget = forget

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        return not_loaded_answer(self._deferred, tool, tool_context)

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        self._forget(invocation_context.invocation_id)
