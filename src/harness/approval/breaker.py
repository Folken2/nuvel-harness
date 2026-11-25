"""Stops asking after the person has said no several times in a row.

A model whose call was denied sometimes tries the same thing another way, and the person
is asked again, and again. Each card is theirs to answer, so past a few refusals in a row
the gate stops asking: the denied call's answer tells the model to make no more calls and
talk to the person, and any call that would ask after that is refused without a card.

Counted from the run's own events, within the person's current message: an approval ends
the row, and so does their next message, which starts a new invocation. An answer that
decided nothing (they answered another question of the same step) is not a refusal, and
neither is one in their own words: they are talking with the agent, not turning it away,
so it neither adds to the row nor ends it.

Pure. ``gate.py`` asks it at each decision and before each card.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from google.adk.events import Event
from google.adk.tools.tool_confirmation import ToolConfirmation

from .. import wire_names
from ..runs.resume import CONFIRMATION_CALL

STOPPED_NOTE = (
    "The person has now denied several calls in a row, so no more will be put to them in "
    "this turn. Make no further tool calls. Tell them briefly what you were trying to do, "
    "and ask how they would like to go on."
)


def _decision(response: Any) -> bool | None:
    """True for approved, False for denied, None when the answer decided nothing or was
    the person's own words."""
    try:
        confirmation = ToolConfirmation.from_response_dict(response or {})
    except (ValueError, TypeError):
        return None
    if confirmation.confirmed:
        return True
    payload = confirmation.payload
    if isinstance(payload, dict):
        said = payload.get("said")
        if payload.get(wire_names.NOT_DECIDED) or (isinstance(said, str) and said.strip()):
            return None
    return False


def denials_in_a_row(events: Sequence[Event], invocation_id: str) -> int:
    """How many of the person's latest decisions in this invocation were refusals, back
    to their last approval."""
    row = 0
    for event in events:
        if event.invocation_id != invocation_id:
            continue
        for response in event.get_function_responses():
            if response.name != CONFIRMATION_CALL:
                continue
            decided = _decision(response.response)
            if decided is True:
                row = 0
            elif decided is False:
                row += 1
    return row


def stopped(row: int, max_denials: int) -> bool:
    """Whether the gate has stopped asking. A limit of 0 never stops."""
    return max_denials > 0 and row >= max_denials
