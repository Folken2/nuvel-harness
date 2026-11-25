"""An answer to one wait answers the others the run holds, so the run can go on.

One model step can leave the run waiting on more than one thing: a call the gate holds
for approval (./gate.py) and a question (tools/ask.py), say. The person answers one,
and ADK, given a partial answer, keeps waiting for the rest; but its check for "still
paused" only looks at the last two events, so it marks the agent ended instead, with
no model call, and the turn fails (seen in production).

So the person's answer carries a result for every other wait still open in that run:
a held call wasn't decided, so it doesn't run (the gate reads NOT_DECIDED); a
question wasn't answered. The model reads both and decides what to ask next. A call
still runs at most once: one that wasn't decided never ran.
"""

from __future__ import annotations

from typing import Any, Optional

from google.adk.events import Event
from google.genai import types

from .. import wire_names
from ..runs.resume import CONFIRMATION_CALL
from ..tools.ask import TOOL_NAME as ASK_TOOL_NAME

# wire_names.NOT_DECIDED is in the confirmation's payload when the person answered
# something else (gate.py reads it).
NOT_ANSWERED = {
    "status": "not_answered",
    "note": (
        "The person decided on the approval you asked for at the same time and didn't"
        " answer this question. Ask it again if you still need the answer."
    ),
}


def _result_for(call: types.FunctionCall) -> dict[str, Any]:
    if call.name == CONFIRMATION_CALL:
        return {"confirmed": False, "payload": {wire_names.NOT_DECIDED: True}}
    if call.name == ASK_TOOL_NAME:
        return dict(NOT_ANSWERED)
    return {"status": "not_answered", "note": "The person answered something else first. Ask again if it's still needed."}


def other_waits(events: list[Event], message: types.Content) -> list[types.FunctionCall]:
    """The calls still waiting on the person in the run `message` answers, apart from
    the ones it answers. Empty when the message answers nothing, or everything."""
    answering = {
        part.function_response.id
        for part in message.parts or []
        if part.function_response and part.function_response.id
    }
    if not answering:
        return []
    calls = {call.id: (event, call) for event in events for call in event.get_function_calls() if call.id}
    invocations = {calls[call_id][0].invocation_id for call_id in answering if call_id in calls}
    if len(invocations) != 1:
        return []
    [invocation] = invocations
    answered = {r.id for event in events for r in event.get_function_responses()} | answering
    open_ids = [
        call_id
        for event in events
        if event.invocation_id == invocation
        for call_id in sorted(event.long_running_tool_ids or ())
        if call_id not in answered and call_id in calls
    ]
    return [calls[call_id][1] for call_id in dict.fromkeys(open_ids)]


def with_other_waits_answered(events: list[Event], message: types.Content) -> list[types.Content]:
    """What the person's answer becomes, in order, or [] when it stays as it is.

    Two messages when it holds approvals and other answers: ADK reads approvals only
    from the last message the person sent, and keeps any message holding one out of
    the model's view, so the other answers go first, in a message of their own."""
    waits = other_waits(events, message)
    if not waits:
        return []
    responses = [part for part in message.parts or [] if part.function_response] + [
        types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=_result_for(call)))
        for call in waits
    ]
    others = [part for part in message.parts or [] if not part.function_response]
    approvals = [part for part in responses if part.function_response.name == CONFIRMATION_CALL]
    answers = [part for part in responses if part.function_response.name != CONFIRMATION_CALL]
    role = message.role or "user"
    if not approvals or not answers:
        return [types.Content(role=role, parts=[*others, *responses])]
    return [types.Content(role=role, parts=[*others, *answers]), types.Content(role=role, parts=approvals)]


def not_decided(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get(wire_names.NOT_DECIDED) is True
