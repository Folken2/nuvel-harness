"""Tells whoever reads a run that one of its model calls is now in the ledger.

The composer's meters must match what the ledger holds, so the web app reads the
person's buckets again when this frame arrives, not when the model's event does: the
ledger write commits in its own task, after that event. The frame carries nothing the
web app would trust over the database; it only says when to read.

Like the inbox (./steering.py), the driver sets where frames go around a run, and the
tasks the run starts (the ledger's writes among them) copy the context.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Callable

from .. import wire_names

_sink: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar(
    "usage_frame_sink", default=None
)


def usage_frame() -> str:
    """The frame, in /run_sse's own form."""
    return f"data: {json.dumps({wire_names.USAGE_FRAME: {'recorded': True}})}\n\n"


def set_frame_sink(sink: Callable[[str], None]) -> contextvars.Token[Callable[[str], None] | None]:
    """Around a run, by the driver: where this run's frames go."""
    return _sink.set(sink)


def reset_frame_sink(token: contextvars.Token[Callable[[str], None] | None]) -> None:
    _sink.reset(token)


def current_announcer() -> Callable[[], None] | None:
    """What says "recorded" to the run this code is part of; None outside a driven run.

    Taken when the run starts, so a write finishing later still reaches its own run.
    """
    sink = _sink.get()
    if sink is None:
        return None
    return lambda: sink(usage_frame())
