"""Puts an interrupted run back together from the session store, so a resume is safe.

A tool call runs at most once. We can't tell "never started" from "finished but not
written", and a connected app can send an email or move money, so a call with no
written result is answered as interrupted, and the model decides what to do. ADK's own
resume would run it again.

The one exception is a tool the product names as safe to run again: it only reads, so a
second run changes nothing. Its answer says so and asks for the call again, and the
model makes it, instead of working out whether it may. The call
still isn't run here: an answer is all a resume writes.

Everything goes through here because plain ADK resume is wrong in the other cases too:
it calls the model again for a run that already ended, and with an unanswered call and
a stateDelta (which the agent requires) it pauses and silently does nothing.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from google.adk.events import Event
from google.adk.sessions import BaseSessionService, Session
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types

from .own_messages import is_own_message
from .regenerate import is_user_message
from .. import wire_names
from .registry import Run, RunKey, RunRegistry, RunStatus
from .steering import frame_of_event

INTERRUPTED_RESULT = {
    "status": "interrupted",
    "note": (
        "The service restarted while this tool was running. Its outcome is unknown: "
        "it may or may not have taken effect. Check before repeating it."
    ),
}


# For a call to a tool that only reads: nothing to check, so the model is asked to repeat it.
REPEAT_RESULT = {
    "status": "interrupted",
    "note": (
        "The service restarted while this tool was running, so its result was lost. This "
        "tool only reads and changes nothing: call it again with the same arguments."
    ),
}


# The call ADK makes to ask the person to approve another (approval/gate.py).
CONFIRMATION_CALL = "adk_request_confirmation"


def is_persons_answer(event: Event) -> bool:
    """The person answering a question (tools/ask.py): it starts a run of its own, in
    the invocation that asked. A tool's own result is never the person's."""
    return event.author == "user" and bool(event.get_function_responses())


def starts_a_run(event: Event) -> bool:
    return is_user_message(event) or is_persons_answer(event)


def _confirmed(response: Any) -> bool:
    try:
        return ToolConfirmation.from_response_dict(response or {}).confirmed
    except (ValueError, TypeError):
        return False


def approved_calls(events: list[Event], answer: Event) -> list[types.FunctionCall]:
    """The calls the person approved in `answer`. ADK runs one when the run that the
    approval starts makes its first model call, so a crash can leave it unanswered in
    that run although the call itself was made in the run before."""
    approved = {
        response.id
        for response in answer.get_function_responses()
        if response.name == CONFIRMATION_CALL and _confirmed(response.response)
    }
    calls: list[types.FunctionCall] = []
    for event in events:
        for call in event.get_function_calls():
            original = (call.args or {}).get("originalFunctionCall")
            if call.name != CONFIRMATION_CALL or call.id not in approved or not isinstance(original, dict):
                continue
            if original.get("id") and original.get("name"):
                calls.append(
                    types.FunctionCall(id=original["id"], name=original["name"], args=original.get("args") or {})
                )
    return calls


@dataclass
class RunHistory:
    invocation_id: str
    events: list[Event]
    # Approved in the answer that started this run, and run by it (approved_calls).
    approved: list[types.FunctionCall] = field(default_factory=list)

    @property
    def ended(self) -> bool:
        """Nothing to resume: ADK would call the model again. The model's last word ends
        a run even without ADK's end marker, which a long-running tool that answered at
        once leaves out (tools/ask.py), and so does a question still waiting: only the
        person's answer goes on from there."""
        last = self.events[-1]
        return (
            any(event.actions.end_of_agent for event in self.events)
            or bool(last.error_code)
            or (last.author != "user" and bool(last.content) and last.is_final_response())
        )

    @property
    def steps(self) -> int:
        """Completed work, so a crash loop can be told from a run that's getting on.
        Answers written here don't count, and nor does the event each resume writes for
        its state, which has no content, nor a note the agent wrote into the history
        (own_messages.py): they aren't progress."""
        return sum(
            1
            for event in self.events
            if event.content
            and not starts_a_run(event)
            and not is_own_message(event)
            and not (event.custom_metadata or {}).get(wire_names.INTERRUPTED)
        )

    def unanswered_calls(self) -> list[types.FunctionCall]:
        """Without the ones waiting for a person: that's a pause, not a crash."""
        answered = {r.id for event in self.events for r in event.get_function_responses()}
        waiting = {
            call_id for event in self.events for call_id in event.long_running_tool_ids or ()
        }
        calls = [
            call
            for event in self.events
            for call in event.get_function_calls()
            if call.id not in answered and call.id not in waiting
        ]
        # ADK would run an approved call again: the approval is still the last word.
        return calls + [call for call in self.approved if call.id not in answered]


def read_history(events: list[Event], since: float) -> RunHistory | None:
    """The run started at `since`: one run per session, so it's the first message, or
    answer, after that. None when it never got there, so nothing has happened. An answer
    continues the invocation that asked, so the run is only what came after it: the
    question's part was the run before, already saved."""
    for index, event in enumerate(events):
        if event.timestamp >= since and starts_a_run(event):
            invocation_id = event.invocation_id
            return RunHistory(
                invocation_id,
                [e for e in events[index:] if e.invocation_id == invocation_id],
                [call for answer in _answers_from(events, index) for call in approved_calls(events[:index], answer)],
            )
    return None


def _answers_from(events: list[Event], index: int) -> list[Event]:
    """The person's answer at `index`, which can be more than one event when it answered
    a step's other waits too (approval/other_waits.py): its approvals come last."""
    answers: list[Event] = []
    for event in events[index:]:
        if not is_persons_answer(event) or event.invocation_id != events[index].invocation_id:
            break
        answers.append(event)
    return answers


async def answer_interrupted_calls(
    sessions: BaseSessionService,
    session: Session,
    history: RunHistory,
    author: str,
    safe_to_repeat: Collection[str] = (),
) -> None:
    """``safe_to_repeat`` names the tools whose interrupted call the model is asked to
    make again; every other call's outcome is unknown, and it is told to check."""
    calls = history.unanswered_calls()
    if not calls:
        return
    event = Event(
        invocation_id=history.invocation_id,
        author=author,
        custom_metadata={wire_names.INTERRUPTED: True},
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id,
                        name=call.name,
                        response=REPEAT_RESULT if call.name in safe_to_repeat else INTERRUPTED_RESULT,
                    )
                )
                for call in calls
            ],
        ),
    )
    await sessions.append_event(session, event)
    history.events.append(event)


def past_frames(history: RunHistory) -> list[str]:
    """What a reader gets for the part already done, in /run_sse's own form. Text that
    was streaming at the crash isn't here: it was never saved, and the model writes it
    again."""
    frames: list[str] = []
    for event in history.events:
        # The person's answer that started this run is theirs, not the run's.
        # Nor is a note the agent wrote for the model part of the transcript.
        if is_persons_answer(event) or is_own_message(event):
            continue
        if not is_user_message(event):
            frames.append(f"data: {event.model_dump_json(exclude_none=True, by_alias=True)}\n\n")
        # A message the person sent while it ran (./steering.py), where it went in.
        elif (frame := frame_of_event(event)) is not None:
            frames.append(frame)
    return frames


@dataclass(frozen=True)
class NotResumed:
    """The run isn't back: there's nothing of it in the session ("nothing_to_resume"),
    or it has done nothing since the last resume, which is a crash loop ("no_progress")."""

    error: str


@dataclass(frozen=True)
class AlreadyEnded:
    """It had finished and only its last frames were lost: the run is back, ended."""

    steps: int


@dataclass(frozen=True)
class ToResume:
    """The run is back with what it had done, and is the caller's to drive with `body`."""

    steps: int
    body: bytes
    # The session's state as the resume found it.
    state: Mapping[str, Any]


async def restore_to_resume(
    registry: RunRegistry,
    sessions: BaseSessionService,
    key: RunKey,
    since: float,
    after_steps: int,
    body: dict[str, Any],
    safe_to_repeat: Collection[str] = (),
) -> tuple[Run, NotResumed | AlreadyEnded | ToResume]:
    """Puts the run started at `since` back in the registry after a restart, its
    interrupted calls answered, ready to be driven with the body of the call that
    started it. `after_steps` is how far it had got the last time it was resumed.

    Raises RunActive when the session's run is already back. Until the caller's driver
    has it, an error here ends the run failed and forgets it (registry.until_driven)."""
    app_name, user_id, session_id = key
    # First, so that two callers can't both write the answers below.
    run = registry.restore(key, since)

    with registry.until_driven(key, run, restored=True):
        session = await sessions.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        history = read_history(session.events, since) if session else None
        if history is None:
            registry.forget(key, run)
            return run, NotResumed("nothing_to_resume")
        if not history.ended and history.steps <= after_steps:
            registry.forget(key, run)
            return run, NotResumed("no_progress")

        if history.ended:
            # It finished and only its last frames were lost.
            for frame in past_frames(history):
                run.add_frame(frame)
            run.end(RunStatus.FINISHED)
            return run, AlreadyEnded(history.steps)

        steps = history.steps
        await answer_interrupted_calls(
            sessions, session, history, author=app_name, safe_to_repeat=safe_to_repeat
        )
        for frame in past_frames(history):
            run.add_frame(frame)
        body.pop("newMessage", None)
        body["invocationId"] = history.invocation_id
        resumed = json.dumps(body).encode()
    return run, ToResume(steps, resumed, session.state)
