"""What a background task has done so far, read from its run's frames (./tasks.py).

Pure: frames in, progress out. A task's report is its last message without a tool
call, which is what the model is told to make its report (prompt.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..replies.agent_note import NOTE_CALL

# What a report keeps. The parent reads it into its own context, so it's bounded.
RESULT_MAX_CHARS = 20_000
TRUNCATED_NOTE = "\n\n[The report was cut here: it was too long.]"


@dataclass
class TaskProgress:
    # Tool calls the task has finished.
    steps: int = 0
    # The tool it's running now, by name; None while the model thinks or writes.
    activity: str | None = None
    # Its latest message that called no tool: the report once the run has ended.
    result: str = ""
    error: str | None = None

    def apply(self, frame: str) -> bool:
        """Takes one frame in. Whether anything a reader shows changed."""
        event = _event_of(frame)
        if event is None or event.get("partial"):
            return False
        before = (self.steps, self.activity, bool(self.result), self.error)
        if event.get("errorCode") or event.get("errorMessage"):
            self.error = str(event.get("errorMessage") or event.get("errorCode"))
        parts = (event.get("content") or {}).get("parts") or []
        # A note after a bad reply is the agent's own call, no step of the task.
        calls = [
            p["functionCall"]
            for p in parts
            if isinstance(p.get("functionCall"), dict) and p["functionCall"].get("name") != NOTE_CALL
        ]
        answered = [
            p
            for p in parts
            if isinstance(p.get("functionResponse"), dict) and p["functionResponse"].get("name") != NOTE_CALL
        ]
        if calls:
            self.activity = str(calls[-1].get("name") or "") or None
        if answered:
            self.steps += len(answered)
            self.activity = None
        text = "".join(
            p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought")
        )
        if text.strip() and not calls and event.get("author") != "user":
            self.result = _bounded(text.strip())
        return before != (self.steps, self.activity, bool(self.result), self.error)


def _event_of(frame: str) -> dict | None:
    data = "\n".join(
        line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")
    )
    if not data:
        return None
    try:
        event = json.loads(data)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _bounded(text: str) -> str:
    if len(text) <= RESULT_MAX_CHARS:
        return text
    return text[: RESULT_MAX_CHARS - len(TRUNCATED_NOTE)] + TRUNCATED_NOTE
