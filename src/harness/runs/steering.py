"""Messages that join a run while it works: the person's "Send now".

A run has an inbox. The web app puts a message in it, and before each model call the
run takes what's there and writes it into its own history (plugins/steering.py). It
has to be the run that writes it: the runner writes to the same session object, and
a second writer would race it.

Each message becomes an ordinary person's event of the same invocation, tagged with
the web app's id for it, so the next steps, the next turns and a reload all see it
where it went in. The frame a run sends when it takes one tells the web app which
messages went in: one that's still in the inbox when the run ends never did, and the
web app's queue still has it.
"""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from google.adk.events import Event
from google.adk.sessions import BaseSessionService, Session
from google.genai import types

from .. import wire_names

logger = logging.getLogger(__name__)

# The Postgres store reads events back by timestamp, and ADK stamps the model's reply
# before the callback that writes this one runs. Just after the session's last event
# is before that reply, which was stamped after that event was written to the store.
AFTER_LAST_EVENT_SECONDS = 1e-6

_inbox: contextvars.ContextVar[RunInbox | None] = contextvars.ContextVar(
    "run_inbox", default=None
)


@dataclass(frozen=True)
class Steer:
    """One message, under the id the web app's queue knows it by."""

    id: str
    text: str


class RunInbox:
    """What's waiting to join one run. Lives and dies with the run."""

    def __init__(self, on_taken: Callable[[Steer], None]) -> None:
        self._waiting: list[Steer] = []
        self._on_taken = on_taken
        self._closed = False

    def put(self, steer: Steer) -> bool:
        """False once the run has ended: it would never be taken."""
        if self._closed:
            return False
        if all(waiting.id != steer.id for waiting in self._waiting):
            self._waiting.append(steer)
        return True

    def take(self) -> list[Steer]:
        """What's waiting, once. Each is `went_in` when it's in the history."""
        taken, self._waiting = self._waiting, []
        return taken

    def went_in(self, steer: Steer) -> None:
        self._on_taken(steer)

    def close(self) -> None:
        """The run has ended. What's left stays in the web app's queue."""
        self._closed = True
        if self._waiting:
            logger.info("%d message(s) didn't join before the run ended", len(self._waiting))
        self._waiting = []


def current_inbox() -> RunInbox | None:
    """The inbox of the run this code is part of; None outside a run."""
    return _inbox.get()


def set_current_inbox(inbox: RunInbox) -> contextvars.Token[RunInbox | None]:
    """Around a run, by the driver. The tasks ADK starts copy its context."""
    return _inbox.set(inbox)


def reset_current_inbox(token: contextvars.Token[RunInbox | None]) -> None:
    _inbox.reset(token)


def steer_event(steer: Steer, invocation_id: str, last_event: Event | None) -> Event:
    event = Event(
        invocation_id=invocation_id,
        author="user",
        content=types.Content(role="user", parts=[types.Part(text=steer.text)]),
        custom_metadata={wire_names.STEER_ID: steer.id},
    )
    if last_event is not None:
        event.timestamp = last_event.timestamp + AFTER_LAST_EVENT_SECONDS
    return event


async def write_steers(
    inbox: RunInbox, session: Session, sessions: BaseSessionService, invocation_id: str
) -> list[types.Content]:
    """Takes what's waiting and writes each into the run's history, as the run. Returns
    what was written, for the model call about to be made: its request was built from
    the history before these went in. ``session`` has to be the object the runner
    writes to (see the module docstring)."""
    written: list[types.Content] = []
    for steer in inbox.take():
        event = steer_event(steer, invocation_id, session.events[-1] if session.events else None)
        await sessions.append_event(session, event)
        if event.content:
            written.append(event.content)
        inbox.went_in(steer)
    return written


def steer_id_of(event: Event) -> str | None:
    """The web app's id for a message that joined a run, else None."""
    steer_id = (event.custom_metadata or {}).get(wire_names.STEER_ID)
    return steer_id if isinstance(steer_id, str) else None


def steer_frame(steer: Steer) -> str:
    """What a reader gets when the run takes a message, in /run_sse's own form."""
    return f"data: {json.dumps({wire_names.STEER_FRAME: {'id': steer.id, 'text': steer.text}})}\n\n"


def frame_of_event(event: Event) -> str | None:
    """The same frame, for a message read back from the session after a restart."""
    steer_id = steer_id_of(event)
    if steer_id is None or not event.content or not event.content.parts:
        return None
    text = "".join(part.text or "" for part in event.content.parts)
    return steer_frame(Steer(id=steer_id, text=text))
