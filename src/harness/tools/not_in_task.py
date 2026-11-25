"""Whether a run is a background task's (runs/tasks.py), and the tools one doesn't get."""

from __future__ import annotations

from typing import Any, Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext

from .. import wire_names


def in_task(state: Any) -> bool:
    return bool(state and state.get(wire_names.TASK_MARK))


class NotInTaskToolset(BaseToolset):
    """Tools a background task doesn't get, such as asking the person: nobody is there."""

    def __init__(self, tools: list[BaseTool] | BaseToolset) -> None:
        super().__init__()
        self._tools = tools

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        if readonly_context is not None and in_task(readonly_context.state):
            return []
        if isinstance(self._tools, BaseToolset):
            return await self._tools.get_tools(readonly_context)
        return list(self._tools)

    async def process_llm_request(self, *, tool_context: ToolContext, llm_request: LlmRequest) -> None:
        """What the toolset adds to the request besides its tools, on the same condition."""
        if isinstance(self._tools, BaseToolset) and not in_task(tool_context.state):
            await self._tools.process_llm_request(tool_context=tool_context, llm_request=llm_request)

    async def close(self) -> None:
        if isinstance(self._tools, BaseToolset):
            await self._tools.close()
