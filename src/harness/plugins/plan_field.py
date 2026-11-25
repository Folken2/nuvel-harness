"""Lets the agent's own tool calls carry the plan: the callbacks only.

What gets the field, and how it becomes a todo_list call, is tools/plan_field.py.
The plan runs even when the call that carried it is held for approval or refused: it then
shows that step in progress, and the model corrects it with its next response.
"""

from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from .. import run_state
from ..tools.plan_field import add_field, carriers, hide_plan_tool, lift, plan_tool_hidden


class PlanFieldPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="plan_field")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        tools = llm_request.config.tools if llm_request.config and llm_request.config.tools else []
        # Only the tools that took the field: one with a todos of its own keeps it.
        names = add_field(tools, carriers(llm_request.tools_dict))
        if plan_tool_hidden(llm_request.tools_dict, names):
            hide_plan_tool(tools)
        # The response callback has no request: it reads which tools took the field here.
        callback_context.state[run_state.PLAN_FIELD_TOOLS] = sorted(names)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        # A partial response's calls never run; the final one carries them all.
        if llm_response.partial:
            return None
        names = set(callback_context.state.get(run_state.PLAN_FIELD_TOOLS) or [])
        # In place, returning nothing: a returned response would end the chain, and the
        # ledger, the cache count and the logs after this one must still see the call.
        lift(llm_response.content, names)
        return None
