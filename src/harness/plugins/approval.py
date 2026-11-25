"""Asks the person before a tool with effects runs: the callback only.

What asks, and how, is approval/gate.py. A plugin, not the agent's own callback, so
it holds for every tool, the person's connected apps included.
"""

from __future__ import annotations

from typing import Any, Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ..approval.gate import ApprovalGate
from ..approval.other_waits import with_other_waits_answered


class ApprovalPlugin(BasePlugin):
    def __init__(self, gate: ApprovalGate) -> None:
        super().__init__(name="approval")
        self.gate = gate

    async def on_user_message_callback(
        self, *, invocation_context: InvocationContext, user_message: types.Content
    ) -> Optional[types.Content]:
        """An answer to a held call or a question answers the step's other waits too
        (approval/other_waits.py). All but the last message go into the session here;
        the runner saves the last as the person's."""
        messages = with_other_waits_answered(invocation_context.session.events, user_message)
        if not messages:
            return None
        for content in messages[:-1]:
            await invocation_context.session_service.append_event(
                invocation_context.session,
                Event(invocation_id=invocation_context.invocation_id, author="user", content=content),
            )
        return messages[-1]

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        return await self.gate.check(tool, tool_args, tool_context)
