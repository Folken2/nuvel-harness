"""Records each tool call of a run in the usage ledger (logic in ``ledger/tool_meter.py``)."""

from __future__ import annotations

from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from ..ledger import tool_meter


class ToolLedgerPlugin(BasePlugin):
    """One operation per tool call that runs.

    Goes right after the approval plugin, the one before-tool callback that answers
    a call itself: a call it holds or denies never reaches this one, so it isn't
    recorded as having run. And before the tool result limit, the first after-tool
    callback that returns a value and so ends that chain.
    """

    def __init__(self) -> None:
        super().__init__(name="tool_ledger")

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        tool_meter.begin(tool, tool_context.function_call_id)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        tool_meter.finish(tool_context.function_call_id, result)
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict[str, Any]]:
        tool_meter.failed(tool_context.function_call_id, tool, error)
        return None
