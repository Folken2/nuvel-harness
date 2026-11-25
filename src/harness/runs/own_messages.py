"""Messages the agent writes into a run's history itself, in the user's turn.

A provider reads the conversation as turns, and something the model should know from
here on (which files the conversation has) is told most cheaply as one more message
after the person's: the prompt before it stays as it was. Such a message is saved with
the session so every later request sends it again in the same place, and marked, so
nothing that looks for the person's words (a regenerate, a resume, a summary's
carried messages) takes it for theirs. It is never sent to the web app: it isn't part of
the transcript.
"""

from __future__ import annotations

from typing import Optional

from google.adk.events import Event
from google.genai import types

from .. import wire_names


def own_message(invocation_id: str, text: str, last_event: Optional[Event]) -> Event:
    """The message as the session keeps it, placed after ``last_event``."""
    event = Event(
        invocation_id=invocation_id,
        author="user",
        content=types.Content(role="user", parts=[types.Part(text=text)]),
        custom_metadata={wire_names.OWN_MESSAGE: True},
    )
    if last_event is not None and event.timestamp <= last_event.timestamp:
        event.timestamp = last_event.timestamp + 0.001
    return event


def is_own_message(event: Event) -> bool:
    return bool((event.custom_metadata or {}).get(wire_names.OWN_MESSAGE))
