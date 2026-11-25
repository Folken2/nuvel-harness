"""The files a built-in ability made in this run (tools/image.py, tools/research.py).

Kept in the run's ``temp:`` state, so it lasts the invocation and is never saved.
``files/artifacts.py`` reads it: a file made a moment ago isn't listed back to the
model, which already has what it needs from the tool's result.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from .. import run_state


def made_in_run(state: Any) -> list[str]:
    names = state.get(run_state.MADE_FILES)
    return [name for name in names if isinstance(name, str)] if isinstance(names, list) else []


def note_made(tool_context: ToolContext, name: str) -> None:
    """Records ``name`` as made by this run."""
    made = made_in_run(tool_context.state)
    if name not in made:
        tool_context.state[run_state.MADE_FILES] = [*made, name]
