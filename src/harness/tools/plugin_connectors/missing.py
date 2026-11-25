"""What a turn goes without from the person's plugins, and how the model is told."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

NOT_LOADED = "it couldn't be loaded"


@dataclass(frozen=True)
class Missing:
    plugin: str
    # None: the whole plugin, skills included.
    connector: str | None
    # Words that finish "... isn't available:", such as "it couldn't be reached".
    reason: str


def notice(missing: Sequence[Missing], closing: str) -> str:
    """The lines added to the turn's instruction, so the model can say what's missing
    rather than look for tools that aren't there. `closing` is the caller's own
    paragraph after the list: what to do about it, and where the person can't fix it."""
    lines = [
        f'- The "{m.connector}" connector of the "{m.plugin}" plugin: {m.reason}.'
        if m.connector is not None
        else f'- The "{m.plugin}" plugin: {m.reason}.'
        for m in missing
    ]
    return (
        "Parts of this person's plugins aren't available in this conversation right now:\n"
        + "\n".join(lines)
        + "\n"
        + closing
    )
