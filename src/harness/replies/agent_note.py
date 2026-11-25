"""A note to the model after a bad reply, so the run goes on instead of ending.

ADK's loop ends on a reply with no tool call in it, and a callback can only change the
reply, not ask the model again. So the reply gets one call of the agent's own,
``agent_note``, that the plugin answers with the note, and the loop goes on.

The provider never sees that call. Before each model call ``as_sent`` takes it out of the
request, and its answer becomes the note as a plain user turn. A tool call the model
never made would need its own reasoning in front of it (Anthropic with thinking on) or a
signature (Gemini), so the request never carries one. The session keeps the call,
because the loop needs it there. The web app and a task's progress leave it out by its
name (``runs/task_progress.py``).
"""

from __future__ import annotations

import uuid
from typing import Any

from google.adk.models import LlmResponse
from google.genai import types

__all__ = ["add_note"]

NOTE_CALL = "agent_note"


def add_note(response: LlmResponse, kind: str) -> None:
    """In place: the reply keeps what it has, followed by the note call for ``kind``."""
    parts = list(response.content.parts or []) if response.content else []
    call = types.FunctionCall(id=f"call-{uuid.uuid4().hex}", name=NOTE_CALL, args={"kind": kind})
    response.content = types.Content(role="model", parts=[*parts, types.Part(function_call=call)])


def note_result(note: str) -> dict[str, Any]:
    return {"note": note}


def as_sent(contents: list[types.Content]) -> list[types.Content]:
    """The request's history as the provider gets it: no note calls, and each note's
    answer as user text. Contents without a note are the same objects."""
    sent: list[types.Content] = []
    for content in contents:
        parts = content.parts or []
        if not any(_is_note(part) for part in parts):
            sent.append(content)
            continue
        answers_note = any(
            part.function_response is not None and part.function_response.name == NOTE_CALL for part in parts
        )
        kept: list[types.Part] = []
        for part in parts:
            if part.function_call is not None and part.function_call.name == NOTE_CALL:
                continue
            response = part.function_response
            if response is not None and response.name == NOTE_CALL:
                note = (response.response or {}).get("note")
                if isinstance(note, str) and note:
                    kept.append(types.Part(text=note))
                continue
            kept.append(part)
        # A model turn left with reasoning alone (an empty reply's) goes: there is nothing in
        # it to reason about any more.
        if kept and not all(part.thought for part in kept):
            # A note's answer comes from the agent, so it goes as the user's turn.
            sent.append(types.Content(role="user" if answers_note else content.role, parts=kept))
    return sent


def _is_note(part: types.Part) -> bool:
    return (part.function_call is not None and part.function_call.name == NOTE_CALL) or (
        part.function_response is not None and part.function_response.name == NOTE_CALL
    )
