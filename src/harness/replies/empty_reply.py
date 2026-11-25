"""An empty reply: the model is told once to go on.

A reply with no text and no tool call (or reasoning alone) ends ADK's loop, and the
turn then fails with nothing to show. Once per run it gets a note instead
(``agent_note.py``), so the model is asked again (OpenHands,
``openhands/sdk/agent/response_dispatch.py``). A second empty reply in the same run is
left alone: it ends the turn as it did before.
"""

from __future__ import annotations

from google.adk.models import LlmResponse
from google.genai import types

KIND = "empty_reply"

NOTE = (
    "Your last reply was empty: it had no text and no tool call. Go on with the task: "
    "answer the user, or make the next tool call."
)


def is_empty(response: LlmResponse) -> bool:
    """A finished reply with nothing in it: no tool call and no text outside reasoning."""
    if response.partial or response.error_code:
        return False
    if response.finish_reason not in (None, types.FinishReason.STOP):
        return False
    parts = response.content.parts if response.content else None
    for part in parts or []:
        if part.text is not None and part.thought:
            continue
        if part.text is not None and not part.text.strip():
            continue
        # Text, a call, a file, code: something to go on from.
        return False
    return True
