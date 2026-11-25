"""Tells the model when its calls repeat, then pauses the run (runs/loop_guard.py).

Two plugins, because their places in the order differ. Neither keeps anything between
callbacks: each reads the run's steps from its events.

``LoopGuardPlugin``, after ``BadRepliesPlugin`` (which repairs a call's arguments first and
answers the note call) and before the approval gate (so a call that won't run is never
asked about):

- after a reply, when one of its calls makes ``note_at`` repeats, the reply gets the
  agent's note call (replies/agent_note.py), so the note arrives with that call's result.
  In place, returning nothing;
- before a tool, a call that would make ``stop_at`` repeats is held: its answer tells the
  model to ask the person, with the tool named here, whether to go on. Their answer
  starts the count again. In a run that can't ask (a background task, or no such tool)
  the call is refused instead.

``LoopPausePlugin``, right after the run limits and before anything that prepares the
next model call (a message the person sent to the running turn is taken there, and must
not be used up by a call that is never made): after such a refusal, where nobody could
be asked, the run ends paused, with the reason ``loop``, as a run at its ceiling ends.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .. import wire_names
from ..replies import agent_note
from ..runs.ceiling import Pause, paused_response
from ..runs.loop_guard import (
    KIND,
    LoopSettings,
    held_for_the_person,
    refusal,
    repeats,
    repeats_of_call,
    step_for,
    steps_of,
    stopped_at,
)

logger = logging.getLogger(__name__)

# What a pause for repeated calls says in its metadata, beside the ceiling's reasons.
REASON = "loop"


class LoopPausePlugin(BasePlugin):
    """Ends the run paused once a repeated call was refused where nobody could be asked."""

    def __init__(self, settings: LoopSettings) -> None:
        super().__init__(name="loop_pause")
        self._settings = settings

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        times = stopped_at(callback_context.session.events, callback_context.invocation_id)
        if times is None:
            return None
        logger.warning(
            "run %s paused: the same calls repeated %d times", callback_context.invocation_id, times
        )
        return paused_response(Pause(REASON, times, self._settings.stop_at))


class LoopGuardPlugin(BasePlugin):
    """``exempt`` names tools whose repeats never count (a wait, a status check).
    ``ask_tool`` is the tool the model asks the person with; None when the agent has none,
    and then a call that repeats too often pauses the run."""

    def __init__(
        self, settings: LoopSettings, exempt: Collection[str] = (), ask_tool: Optional[str] = None
    ) -> None:
        super().__init__(name="loop_guard")
        self._settings = settings
        self._ask_tool = ask_tool
        # Asking is never a loop: it is how one is ended.
        self._exempt = frozenset({agent_note.NOTE_CALL, *exempt, *([ask_tool] if ask_tool else [])})

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.partial:
            return None
        parts = (llm_response.content.parts if llm_response.content else None) or []
        calls = [part.function_call for part in parts if part.function_call is not None]
        if not calls:
            return None
        steps = steps_of(callback_context.session.events, callback_context.invocation_id, self._exempt)
        for call in calls:
            if (call.name or "") in self._exempt:
                continue
            steps.append(step_for(call))
            if repeats(steps, len(steps) - 1) == self._settings.note_at:
                agent_note.add_note(llm_response, KIND)
                logger.warning(
                    "run %s: %s called the same way %d times; the model is told",
                    callback_context.invocation_id,
                    call.name,
                    self._settings.note_at,
                )
                break
        return None

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        if tool.name in self._exempt:
            return None
        steps = steps_of(tool_context.session.events, tool_context.invocation_id, self._exempt)
        times = repeats_of_call(steps, tool_context.function_call_id)
        if times < self._settings.stop_at:
            return None
        # A background task has nobody to ask (its mark is on its session's state).
        can_ask = self._ask_tool is not None and not tool_context.state.get(wire_names.TASK_MARK)
        logger.warning(
            "run %s: %s held, it would be the same call for the %dth time; %s",
            tool_context.invocation_id,
            tool.name,
            times,
            "the model is told to ask the person" if can_ask else "nobody to ask, so the run pauses",
        )
        if can_ask and self._ask_tool is not None:
            return held_for_the_person(tool.name, times, self._ask_tool)
        return refusal(tool.name, times)

