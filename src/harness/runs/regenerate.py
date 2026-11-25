"""Regenerate: retries the last turn instead of adding a second copy of it.

Every turn from the web app carries the id of its message's row in
``customMetadata``, and ADK stamps it on the stored user event. A regenerate
sends the message again with ``<namespace>_regenerate`` set too. Before that run
starts, the turn it repeats is rewound with ADK's own rewind: its events stay in
the log, but the model and compaction no longer read them, and state and
artifacts go back to how they were. A rewind event marks it.

ADK's public rewind is ``Runner.rewind_async``, which its server offers as
``POST .../sessions/{id}/rewind``. The Runner is the server's own, so the rewind
goes through that route in-process, the way ``driver.py`` drives ``/run_sse``,
with the caller's token, under the same check as any other call.

Only a turn stamped with the same id is rewound. When the last attempt never
reached the agent, the session's last turn is an earlier one, which must stay,
even with the same words.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Optional

import httpx
from google.adk.events import Event, get_active_events

from .. import wire_names
from .own_messages import is_own_message
from .steering import steer_id_of

logger = logging.getLogger(__name__)

ASGIApp = Callable[..., Awaitable[None]]
# What the refusal keeps of a failed rewind, for the web app's log.
DETAIL_CHARS = 300


class RewindFailed(Exception):
    """ADK refused the rewind. The run doesn't start: a retry beats a duplicate."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"rewind answered {status}")
        self.status = status
        self.detail = detail[:DETAIL_CHARS]


def is_user_message(event: Event) -> bool:
    """A person's message, not a rewind, a compaction, a tool's answer or a note the
    agent wrote into the history itself (files/file_list.py)."""
    return (
        event.author == "user"
        and bool(event.content and event.content.parts)
        and not event.get_function_responses()
        and not is_own_message(event)
    )


def turn_to_retry(events: list[Event], message_id: Optional[str]) -> Optional[str]:
    """The invocation id of the last turn, when it answers this same message."""
    if not message_id:
        return None
    for event in reversed(get_active_events(events)):
        # One sent while the turn ran is part of that turn, not its start.
        if is_user_message(event) and steer_id_of(event) is None:
            stamped = (event.custom_metadata or {}).get(wire_names.MESSAGE_ID)
            return event.invocation_id if stamped == message_id else None
    return None


def message_to_retry(body: bytes) -> Optional[str]:
    """The message id a regenerate's run body repeats; None for any other run."""
    # A run can carry 10 MB of attachments: only a regenerate is worth parsing.
    if wire_names.REGENERATE.encode() not in body:
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    # ADK reads either spelling, the alias first.
    metadata = parsed.get("customMetadata", parsed.get("custom_metadata"))
    if not isinstance(metadata, dict) or not metadata.get(wire_names.REGENERATE):
        return None
    message_id = metadata.get(wire_names.MESSAGE_ID)
    return message_id if isinstance(message_id, str) and message_id else None


async def rewind_for_regenerate(
    app: ASGIApp,
    app_name: str,
    user_id: str,
    session_id: str,
    message_id: str,
    headers: Sequence[tuple[bytes, bytes]],
) -> Optional[str]:
    """Rewinds the turn this message started, if it's the session's last. Returns
    the invocation rewound, or None when there's nothing to rewind."""
    session_path = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}"
    transport = httpx.ASGITransport(app=app)
    forwarded = [(name.decode("latin-1"), value.decode("latin-1")) for name, value in headers]
    async with httpx.AsyncClient(
        transport=transport, base_url="http://agent", headers=forwarded
    ) as client:
        found = await client.get(session_path)
        if found.status_code == 404:
            # /run_sse answers a missing session itself, as for any other run.
            return None
        if found.status_code != 200:
            raise RewindFailed(found.status_code, found.text)
        events = [Event.model_validate(event) for event in found.json().get("events", [])]
        invocation_id = turn_to_retry(events, message_id)
        if not invocation_id:
            return None
        rewound = await client.post(
            f"{session_path}/rewind", json={"rewindBeforeInvocationId": invocation_id}
        )
        if rewound.status_code != 200:
            raise RewindFailed(rewound.status_code, rewound.text)
    logger.info("regenerate: rewound %s", invocation_id)
    return invocation_id
