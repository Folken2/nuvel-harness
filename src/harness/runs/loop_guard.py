"""Notices a run that keeps making the same calls and getting the same results.

The step ceiling (ceiling.py) ends a stuck run after dozens of steps; this sees it
after a few. Two shapes count, both by exact match of a call's name and arguments and of
its result:

- the same call again, with the same result every time;
- a short cycle of calls (A, B, A, B) that returns the same results each round.

A different result is progress (a poll whose answer changed, a page that moved on), so
it never counts. A call just made counts as a repeat when the ones before it came back
the same.

Repeating can be what the person wants (ten runs of the same check, a job polled until
it is done), so the guard never decides that for them. At ``note_at`` repeats the model
is told once. At ``stop_at`` the call is held, not refused for good: its answer tells the
model to ask the person whether to go on. Their answer starts the count again, so a yes
lets the same calls run on, and they are asked again only after as many more. Where
nobody can be asked (a background task) the held call ends the run paused instead, like
a run at its ceiling.

Pure, and nothing is remembered: what a run has done is read from its own events at
each step, so a resume after a restart counts the same. The plugin
(``plugins/loop_guard.py``) asks before each tool call and each model call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from google.adk.events import Event
from google.genai import types

__all__ = ["KIND", "NOTE"]

# The kind of note the model gets (replies/agent_note.py).
KIND = "repeated_calls"
NOTE = (
    "You have now made the same tool call, or gone round the same few calls, several "
    "times, with the same arguments and the same result each time. If repeating it is "
    "what the person asked for, go on. If not, another round won't change the result: use "
    "what you have, try a different approach, or tell the person what is in the way. "
    "After a few more rounds the next one will wait for the person's go-ahead."
)
# The key a refused call's result carries, which the next model call reads to stop.
STOPPED = "repeated_call_not_run"
# The longest cycle looked for: A, B, C, A, B, C.
MAX_CYCLE = 3


@dataclass(frozen=True)
class LoopSettings:
    """The model is told at ``note_at`` repeats of a call or a cycle; at ``stop_at`` the
    call is held for the person's go-ahead."""

    note_at: int
    stop_at: int


@dataclass(frozen=True)
class Step:
    """One tool call of a run: what was called, and what came back (None while it
    hasn't)."""

    call_id: Optional[str]
    call: str
    result: Optional[str]

    def same_as(self, other: "Step") -> bool:
        """The same call with the same result. A result not back yet matches any result
        that is back, never another not back: calls made side by side (six identical
        calls in one reply) have shown nothing yet, so they repeat nothing."""
        if self.call != other.call:
            return False
        if self.result is None or other.result is None:
            return self.result is not None or other.result is not None
        return self.result == other.result


def _digest(value: Any) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def step_for(call: types.FunctionCall) -> Step:
    """A call the model has just made, before it runs."""
    return Step(call.id, _digest([call.name, call.args or {}]), None)


def _persons_answer(event: Event) -> bool:
    """The person answering a question or an approval: their word on how to go on."""
    return event.author == "user" and bool(event.get_function_responses())


def steps_of(events: Sequence[Event], invocation_id: str, exempt: Collection[str] = ()) -> list[Step]:
    """The run's tool calls since the person last answered it, in the order the model made
    them, each with its result once it is back. ``exempt`` names tools that are left out:
    waiting on something is their job, and the same answer twice is no loop."""
    steps: list[Step] = []
    by_id: dict[str, int] = {}
    for event in events:
        if event.invocation_id != invocation_id or event.partial:
            continue
        if _persons_answer(event):
            # They have just said how to go on: what came before is no longer counted.
            steps, by_id = [], {}
            continue
        for part in (event.content.parts if event.content else None) or []:
            call, response = part.function_call, part.function_response
            if call is not None and (call.name or "") not in exempt:
                if call.id:
                    by_id[call.id] = len(steps)
                steps.append(step_for(call))
            elif response is not None and response.id in by_id:
                index = by_id[response.id]
                steps[index] = Step(response.id, steps[index].call, _digest(response.response))
    return steps


def repeats(steps: Sequence[Step], index: int) -> int:
    """How many times in a row the call at ``index`` has now been made the same way:
    1 for a call that repeats nothing. A cycle counts its rounds."""
    best = 1
    for period in range(1, MAX_CYCLE + 1):
        rounds = 1
        # Each round is compared with the one after it, so only calls whose results
        # aren't back, the one just made and any beside it, are taken on trust.
        later = steps[index - period + 1 : index + 1]
        while len(later) == period:
            end = index - period * rounds
            earlier = steps[max(end - period + 1, 0) : end + 1] if end >= 0 else []
            if len(earlier) != period or not all(a.same_as(b) for a, b in zip(earlier, later)):
                break
            rounds += 1
            later = earlier
        best = max(best, rounds)
    return best


def repeats_of_call(steps: Sequence[Step], call_id: Optional[str]) -> int:
    """``repeats`` for the call with this id; 1 when it isn't among the steps."""
    for index in range(len(steps) - 1, -1, -1):
        if steps[index].call_id == call_id and call_id is not None:
            return repeats(steps[: index + 1], index)
    return 1


def held_for_the_person(tool_name: str, times: int, ask_tool: str) -> dict[str, Any]:
    """What a call that would repeat once too often is answered with where the person
    can be asked: it hasn't run, and whether it should is theirs to say."""
    return {
        "status": "held",
        "note": (
            f"Not run yet: this would be the same call to {tool_name} for the {times}th time "
            "in a row, with the same result every time so far. Repeating it may be what the "
            f"person wants, and that is for them to say. Call {ask_tool} now: tell them in one "
            "sentence what keeps coming back the same, and ask whether to keep going or stop. "
            "If they say to keep going, make the call again."
        ),
    }


def refusal(tool_name: str, times: int) -> dict[str, Any]:
    """The same, where nobody can be asked: the call isn't run and the run is paused."""
    return {
        "error": (
            f"Not run: this would be the same call to {tool_name} for the {times}th time in a "
            "row, with the same result every time so far. The task is being paused."
        ),
        STOPPED: times,
    }


def stopped_at(events: Sequence[Event], invocation_id: str) -> Optional[int]:
    """The repeat count a refused call of the run's last step carries, if one does."""
    for event in reversed(events):
        if event.invocation_id != invocation_id or event.partial:
            continue
        parts = (event.content.parts if event.content else None) or []
        responses = [part.function_response for part in parts if part.function_response is not None]
        if not responses:
            # Only the step just made counts: an older refusal was already answered.
            if any(part.function_call is not None or part.text for part in parts):
                return None
            continue
        for response in responses:
            times = (response.response or {}).get(STOPPED)
            if isinstance(times, int):
                return times
        return None
    return None
