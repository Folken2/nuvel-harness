"""The safety ceiling: one run's budget, and when it has been reached.

No ADK callbacks and no environment, so it is testable on its own. The plugin
(``plugins/limits.py``) feeds it each model call and tool call, and asks it before
each model call whether the run may go on. What a call or a tool cost is the caller's
to say: this module is handed the functions that price them.

A step is a model call or a tool call, counted together. The
ceiling is reached at "equal or above": a ceiling of 60 steps allows exactly 60.
The ceiling values come with the turn from the web app (``state_keys.LIMITS``),
taken when the run started, so an admin's change applies to the next run only.
Without them the agent's own settings apply.

At the ceiling the run gets one last call: the model is asked to
say what was done and what is left, with tool calls turned off and its reply capped, so
the person reads where things stand and not only that the run stopped. That call spends
past the ceiling, by one short reply. Its reply carries the
pause, and any call after it ends the run with an empty one as before.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from google.adk.events import Event
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from .. import wire_names

# A paused run ends with wire_names.PAUSED in its event's metadata:
# {"reason", "reached", "ceiling"}.


@dataclass(frozen=True)
class Ceiling:
    steps: int
    cost_usd: float
    seconds: float


@dataclass
class RunBudget:
    """What one run has used so far."""

    started_at: float
    steps: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class Pause:
    """Why a run stopped at its ceiling: which value, how much was used, and the ceiling."""

    reason: str  # "steps", "cost" or "time"
    reached: float
    ceiling: float

    def metadata(self) -> dict[str, Any]:
        return {"reason": self.reason, "reached": self.reached, "ceiling": self.ceiling}


def _positive(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def ceiling_from_state(raw: object, fallback: Ceiling) -> Ceiling:
    """The run's ceiling as the web app sent it (``{"ceiling": {"steps", "costUsd",
    "seconds"}}``); each value it didn't send, or sent unusable, is the fallback's."""
    sent = raw.get("ceiling") if isinstance(raw, Mapping) else None
    if not isinstance(sent, Mapping):
        return fallback
    steps = _positive(sent.get("steps"))
    cost = _positive(sent.get("costUsd"))
    seconds = _positive(sent.get("seconds"))
    return Ceiling(
        steps=int(steps) if steps is not None else fallback.steps,
        cost_usd=cost if cost is not None else fallback.cost_usd,
        seconds=seconds if seconds is not None else fallback.seconds,
    )


def reached(budget: RunBudget, ceiling: Ceiling, now: float) -> Optional[Pause]:
    """The pause, if the run may not make another model call; None while it may.

    Steps first, then cost, then time: when several are reached at once the person
    reads the one most likely to make sense of a loop.
    """
    if budget.steps >= ceiling.steps:
        return Pause("steps", budget.steps, ceiling.steps)
    if budget.cost_usd >= ceiling.cost_usd:
        return Pause("cost", round(budget.cost_usd, 6), ceiling.cost_usd)
    elapsed = now - budget.started_at
    if elapsed >= ceiling.seconds:
        return Pause("time", round(elapsed, 3), ceiling.seconds)
    return None


def paused_response(pause: Pause) -> LlmResponse:
    """What the run ends with in place of the model call: nothing for the model to read
    later (an event with no parts isn't sent back to it), and the pause for the web app."""
    return LlmResponse(
        content=types.Content(role="model", parts=[]),
        custom_metadata={wire_names.PAUSED: pause.metadata()},
        turn_complete=True,
    )


LAST_CALL_NOTE = {
    "steps": "the most steps one run may take",
    "cost": "the most one run may cost",
    "time": "the longest one run may go on",
}


def last_call_note(pause: Pause) -> str:
    """What the model is told on the last call: the run is over, and how to close it."""
    limit = LAST_CALL_NOTE.get(pause.reason, "its limit")
    return (
        f"This run has reached {limit}, so it stops here and no more tools can be called. "
        "In a few short sentences, tell the person what has been done so far, what is "
        "left, and what they could ask for next to finish it. Don't start on the work "
        "that is left, and don't apologise at length."
    )


def make_last_call(llm_request: LlmRequest, pause: Pause, max_output_tokens: int) -> None:
    """Turns the request into the run's last call, in place: tool calls off and the
    reply capped. The tools stay declared, so the cached prompt prefix still holds;
    the note is the last thing the model reads."""
    config = llm_request.config or types.GenerateContentConfig()
    llm_request.config = config
    config.tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
    )
    cap = config.max_output_tokens
    config.max_output_tokens = min(cap, max_output_tokens) if cap else max_output_tokens
    llm_request.contents.append(
        types.Content(role="user", parts=[types.Part(text=last_call_note(pause))])
    )


def close_last_call(llm_response: LlmResponse, pause: Pause) -> None:
    """The last call's reply, in place: any tool call it made anyway is dropped, since
    none may run, and the reply carries the pause for the web app."""
    if llm_response.content and llm_response.content.parts:
        llm_response.content.parts = [
            part for part in llm_response.content.parts if part.function_call is None
        ]
    llm_response.custom_metadata = {
        **(llm_response.custom_metadata or {}),
        wire_names.PAUSED: pause.metadata(),
    }


def is_model_call(event: Event) -> bool:
    """A model call that finished, as the run's events keep it: not a piece of a stream."""
    return event.author != "user" and bool(event.usage_metadata) and not event.partial


def _tool_results(events: Sequence[Event]) -> list[types.FunctionResponse]:
    return [
        part.function_response
        for event in events
        for part in (event.content.parts if event.content else None) or []
        if part.function_response is not None
    ]


def spent_before(
    events: Sequence[Event],
    now: float,
    cost_of_call: Callable[[Event], float],
    cost_of_tool: Callable[[types.FunctionResponse], float],
) -> RunBudget:
    """What a run had used before this start, from its own saved events: a resume
    continues with it, so a crash loop can't escape the ceiling. A new run has
    used nothing."""
    calls = [event for event in events if is_model_call(event)]
    tools = _tool_results(events)
    if not calls and not tools:
        return RunBudget(started_at=now)
    # Only what has content: the resume itself writes an event too, and the time the
    # service was down isn't the run's.
    worked = [event for event in events if event.content]
    return RunBudget(
        started_at=now - (worked[-1].timestamp - worked[0].timestamp),
        steps=len(calls) + len(tools),
        cost_usd=sum(cost_of_call(call) for call in calls)
        + sum(cost_of_tool(tool) for tool in tools),
    )


class RunBudgets:
    """Each run in progress: what it has used, and the ceiling it started under."""

    def __init__(self, fallback: Ceiling) -> None:
        self.fallback = fallback
        self._runs: dict[str, tuple[RunBudget, Ceiling]] = {}

    def start(self, run_id: str, budget: RunBudget, limits: object) -> None:
        """``limits`` is what the web app sent with the turn, if anything."""
        self._runs[run_id] = (budget, ceiling_from_state(limits, self.fallback))

    def of(self, run_id: str, now: float, limits: object = None) -> tuple[RunBudget, Ceiling]:
        """The run's budget and ceiling. One that was never started starts here, unused."""
        if run_id not in self._runs:
            self.start(run_id, RunBudget(started_at=now), limits)
        return self._runs[run_id]

    def end(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
