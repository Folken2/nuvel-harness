"""The conversation's file list, told to the model in a message of its own.

ADK's load_artifacts puts the list in the system instruction. The instruction comes
first in a prompt, so each new file changed everything after it for the provider's
cache: the next call paid full price for the whole conversation (seen with
``plugins/prefix_watch.py``). A product whose work is making files
did that at nearly every turn.

So the list goes into the history instead, once each time it changes: a message the
agent writes after the person's own (``runs/own_messages.py``), saved with the session,
so every later request sends it again where it was. Nothing is remembered beside the history. A request
needs a new note when the latest one it holds doesn't say the current list, which also
covers the note having been summarized away or rewound with its turn.

Pure. ``artifacts.py`` reads the files and writes the note.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Optional

from google.genai import types

_OPENING = "[Files in this conversation: "
_NONE_LEFT = "[This conversation has no files any more. Any earlier list is out of date.]"
_NOTE = re.compile(re.escape(_OPENING) + r"(\[.*?\])\. ", re.DOTALL)


def note(names: Sequence[str], loader: str) -> str:
    """What the model reads. ``loader`` is the tool that loads a file by its name."""
    if not names:
        return _NONE_LEFT
    return (
        f"{_OPENING}{json.dumps(sorted(names), ensure_ascii=False)}. This is the full list now, "
        f"in place of any earlier one. To answer a question about one of them, call {loader} "
        "with its name first, even if it was loaded before: don't rely on an earlier answer "
        "about it.]"
    )


def told(contents: Sequence[types.Content]) -> Optional[list[str]]:
    """The list the latest note in a request gives; None when it holds no note."""
    for content in reversed(contents):
        if content.role != "user":
            continue
        for part in reversed(content.parts or []):
            text = part.text or ""
            if text == _NONE_LEFT:
                return []
            found = _NOTE.match(text)
            if found:
                try:
                    names = json.loads(found.group(1))
                except ValueError:
                    continue
                if isinstance(names, list) and all(isinstance(name, str) for name in names):
                    return names
    return None


def needs_telling(contents: Sequence[types.Content], names: Sequence[str]) -> bool:
    """Whether the request should get a note: its latest one says another list, or it
    has none and there are files."""
    said = told(contents)
    return sorted(said or []) != sorted(names)
