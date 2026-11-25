"""The model's bad replies: the callbacks only, logic in ``replies/``.

After each reply, in place, returning nothing, so the ledger, the cache count and the
logs after this plugin see the reply that stands:

- its tool calls' arguments are repaired where the schema leaves no doubt, and each
  repair is logged (``replies/argument_repair.py``);
- a reply cut at the output limit keeps its text, and its calls are held: each is
  answered with a refusal before any other tool callback sees it, so none runs, none is
  asked about and none is recorded as run (``replies/cut_reply.py``). When ADK dropped
  the cut call for its broken arguments, the reply gets a note instead, once per run,
  and a stream's text sent ahead of the dropped call stands, not an error;
- an empty reply gets a note, once per run (``replies/empty_reply.py``).

A note is a call of the agent's own that this plugin answers, so the loop goes on; before
each model call the request is rewritten so the provider sees the note as user text and
never that call (``replies/agent_note.py``).

Right after ``PlanFieldPlugin``: its before-model callback adds the plan field to the
declarations whose schemas this one keeps, and its after-model callback lifts the field
into todo_list calls, which a cut reply's refusal then covers too. Before the approval
plugin, the first other before-tool callback that returns a value.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .. import run_state
from ..replies import agent_note, cut_reply, empty_reply
from ..replies.argument_repair import declared_schemas, repair_arguments

logger = logging.getLogger(__name__)

_NOTES = {empty_reply.KIND: empty_reply.NOTE, cut_reply.DROPPED_KIND: cut_reply.DROPPED_NOTE}


class BadRepliesPlugin(BasePlugin):
    """``notes`` are further kinds of note this plugin answers, by kind: another plugin
    adds the note call to a reply, and the text it stands for is given here."""

    def __init__(self, notes: Optional[Mapping[str, str]] = None) -> None:
        super().__init__(name="bad_replies")
        self._notes = {**(notes or {}), **_NOTES}

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        llm_request.contents = agent_note.as_sent(llm_request.contents)
        tools = llm_request.config.tools if llm_request.config and llm_request.config.tools else []
        callback_context.state[run_state.TOOL_SCHEMAS] = declared_schemas(tools)
        callback_context.state[run_state.CALL_STREAMED] = False
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        # A partial response's calls never run; the final one carries them all. One that
        # writes a call is remembered: a stream that drops it ends with no call in it.
        if llm_response.partial:
            if cut_reply.writes_call(llm_response):
                callback_context.state[run_state.CALL_STREAMED] = True
            return None
        call_streamed = bool(callback_context.state.get(run_state.CALL_STREAMED))
        self._repair(callback_context, llm_response)
        calls = cut_reply.cut_calls(llm_response)
        if calls:
            held = cut_reply.held_ids(calls)
            callback_context.state[run_state.CUT_CALLS] = [
                *(callback_context.state.get(run_state.CUT_CALLS) or []),
                *held,
            ]
            cut_reply.clear_limit_error(llm_response)
            logger.warning(
                "reply cut at the output limit: %d tool call(s) not run (%s)",
                len(held),
                ", ".join(call.name or "?" for call in calls),
            )
        elif cut_reply.text_before_dropped_call(llm_response, call_streamed):
            # The turn goes on past it, so its limit is not the error that ends the turn.
            # Cleared even when this run already sent its note: then the dropped call that
            # follows keeps its own limit error, the turn ends there, and the text stands
            # as the answer so far instead of the turn failing twice.
            cut_reply.clear_limit_error(llm_response)
        elif cut_reply.dropped_call(llm_response, call_streamed):
            if self._first(callback_context, cut_reply.DROPPED_KIND):
                cut_reply.clear_limit_error(llm_response)
                agent_note.add_note(llm_response, cut_reply.DROPPED_KIND)
                logger.warning("reply cut at the output limit, its call dropped: the model is told once")
        elif empty_reply.is_empty(llm_response) and self._first(callback_context, empty_reply.KIND):
            agent_note.add_note(llm_response, empty_reply.KIND)
            logger.warning("empty reply: the model is asked once more")
        return None

    @staticmethod
    def _first(callback_context: CallbackContext, kind: str) -> bool:
        """Whether this run hasn't sent a note of ``kind`` yet; marks it sent."""
        sent = list(callback_context.state.get(run_state.NOTES_SENT) or [])
        if kind in sent:
            return False
        callback_context.state[run_state.NOTES_SENT] = [*sent, kind]
        return True

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        if tool.name == agent_note.NOTE_CALL:
            return agent_note.note_result(self._notes.get(str(tool_args.get("kind")), empty_reply.NOTE))
        if tool_context.function_call_id in (tool_context.state.get(run_state.CUT_CALLS) or []):
            return cut_reply.refusal(tool.name)
        return None

    @staticmethod
    def _repair(callback_context: CallbackContext, llm_response: LlmResponse) -> None:
        schemas = callback_context.state.get(run_state.TOOL_SCHEMAS) or {}
        for part in (llm_response.content.parts if llm_response.content else None) or []:
            call = part.function_call
            schema = schemas.get(call.name) if call else None
            if not call or not isinstance(schema, dict) or not isinstance(call.args, dict):
                continue
            repaired, repairs = repair_arguments(call.args, schema)
            if repairs:
                call.args = repaired
                # Where and what, never the values: they can be the person's data.
                logger.info("repaired %s arguments: %s", call.name, "; ".join(map(str, repairs)))
